import torch
from torch import nn
import torch.nn.functional as F

###########################################
############# differential topK ###########
###########################################
# Calculation of differential topK is based on [Top-K](https://arxiv.org/pdf/2104.03059.pdf), thanks
class PerturbedTopK(nn.Module):
    def __init__(self, k: int, num_samples: int=500, sigma: float=0.05):
        super().__init__()
        self.num_samples = num_samples
        self.sigma = sigma
        self.k = k
    
    def __call__(self, x):
        return PerturbedTopKFuntion.apply(x, self.k, self.num_samples, self.sigma)

class PerturbedTopKFuntion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k: int, num_samples: int=500, sigma: float=0.05):
        # input here is scores with (bs, num_patches)
        b, d = x.shape
        noise = torch.normal(mean=0.0, std=1.0, size=(b, num_samples, d)).to(dtype=x.dtype, device=x.device)
        perturbed_x = x.unsqueeze(1) + noise*sigma # b, nS, d
        topk_results = torch.topk(perturbed_x, k=k, dim=-1, sorted=False)
        indices = topk_results.indices # b, nS, k
        indices = torch.sort(indices, dim=-1).values # b, nS, k

        perturbed_output = F.one_hot(indices, num_classes=d).float() # b, nS, k, d
        indicators = perturbed_output.mean(dim=1) # b, k, d

        # context for backward
        ctx.k = k
        ctx.num_samples = num_samples
        ctx.sigma = sigma

        ctx.perturbed_output = perturbed_output
        ctx.noise = noise

        return indicators
    
    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return tuple([None]*5)
        
        noise_gradient = ctx.noise
        expected_gradient = (
            torch.einsum("bnkd,bnd->bkd", ctx.perturbed_output, noise_gradient)
            / ctx.num_samples
            / ctx.sigma
        )
        grad_input = torch.einsum("bkd,bkd->bd", grad_output, expected_gradient)
        return (grad_input,) + tuple([None]*5)

class PredictorLG(nn.Module):
    """
    """
    def __init__(self, embed_dim=128):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim//2, bias=False),
            nn.GELU()
        )

        self.guidance_conv = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2, bias=False),
            nn.GELU()
        )

        self.out_conv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2, bias=False),
            nn.GELU(),
            # nn.Linear(embed_dim // 2, embed_dim // 4, bias=False),
            # nn.GELU(),
            nn.Linear(embed_dim // 2, 1, bias=False),
            nn.Tanh()
            # nn.Sigmoid()
            # nn.Softmax(dim=-1)
            # nn.LogSoftmax(dim=-1)
        )


    def forward(self, x, guidance):
        '''
        x: shape (bs*n_length, num_tokens, hid_dim) guidance: shape (bs*n_length, 1, hid_dim)
        '''
        x = self.in_conv(x)
        guidance = self.guidance_conv(guidance.repeat(1, x.shape[1], 1))
        local_x = x
        global_x = guidance
        # print("global_x.shape: ", global_x.shape)
        x = torch.cat([local_x, global_x], dim=-1)
        return self.out_conv(x)

class VisualTokenSelection(nn.Module):
    def __init__(self, max_frames=20, embed_dim=128, topk=3):
        super().__init__()
        self.max_frames = max_frames
        self.score_predictor = PredictorLG(embed_dim=embed_dim)
        self.topk_selector = PerturbedTopK(topk)
        self.tanh = nn.Tanh()
    
    def forward(self, x, guidance_frame, guidance_sentence, training=True):
        '''
        x: input embed, shape is (bs, length*Npatch, hid_dim)
        guidance_frame: (bs, frame, hid_dim)
        guidance_sentence: (bs, 1, hid_dim)
        prob = Tanh(MLP(x))
        '''
        
        B, L, D = x.shape
        N = L // self.max_frames
        x = x.reshape(B, -1, N, D) # shape here is (bs, max_frames, n_patches, hid_dim)
        x = x.reshape(-1, N, D) # shape here is (bs*max_frames, n_patches, hid_dim)
        pred_score1 = self.score_predictor(x, guidance_frame.reshape(-1,1,D)).squeeze() # (bs*max_frames, n_patches)
        pred_score2 = self.score_predictor(x, guidance_sentence.repeat(1,self.max_frames,1).reshape(-1,1,D)).squeeze() # (bs*max_frames, n_patches)
        #spatial_pred_score = pred_score1 * self.tanh(pred_score2) # (bs*max_frames, n_patches)
        spatial_pred_score = pred_score1 + pred_score2

        topk_indicator = self.topk_selector(spatial_pred_score) # (bs*max_frames, k, n_patches))

        spatial_x_feature = x # shape here is (bs*max_frames, n_patches, hid_dim)
        selected_patch_feature = torch.einsum("bkl,bld->bkd", topk_indicator, spatial_x_feature) # shape here is (bs*max_frames, topkPatches, hid_dim)

        output = selected_patch_feature.reshape(B, self.max_frames, -1, D) # shape here is (B, max_frames, topkPatches, D)

        return output

class VisualTokenRandomSelection(nn.Module):
    def __init__(self, max_frames, embed_dim=512, topk=3):
        super().__init__()
        self.max_frames = max_frames
        self.topk = topk
    
    def forward(self, x, training=True):
        '''
        x: input embed, shape is (bs, length*Ntokens, hid_dim)
        use cls token as global representation
        prob = Tanh(MLP(x))
        '''
        
        B, L, D = x.shape
        N = L // self.max_frames
        x = x.reshape(B, -1, N, D) # shape here is (bs, max_frames, n_patches, hid_dim)
        x = x.reshape(-1, N, D) # shape here is (bs*max_frames, n_patches, hid_dim)

        # cls token as cls token
        cls_x_feature = x[:, :1, :] # cls_token, shape here is (bs*max_frames, 1, hid_dim)
        # # avg pool of all tokens as cls token
        # cls_x_feature = torch.mean(x, dim=1, keepdim=True)

        spatial_x_feature = x[:, 1:, :] # seperate the cls_token, shape here is (bs*max_frames, n_patches-1, hid_dim)
        patch_len = spatial_x_feature.shape[1]
        selected_indices = torch.randperm(patch_len)[:self.topk].sort()[0]
        selected_patch_feature = spatial_x_feature[:, selected_indices, :]

        output = torch.cat((cls_x_feature, selected_patch_feature), dim=1) # shape here is (bs*max_frames, topkPatches, hid_dim)
        output = output.reshape(B, self.max_frames, -1, D).reshape(B, -1, D) # shape here is (B, max_frames*topkPatches, D) 

        return output

class TextTokenSelection(nn.Module):
    def __init__(self, embed_dim=128, topk=8):
        super().__init__()
        self.score_predictor = PredictorLG(embed_dim=embed_dim)
        self.topk_selector = PerturbedTopK(topk)
        self.tanh = nn.Tanh()
    
    def forward(self, x, guidance_frame, guidance_sentence, training=True):
        '''
        x: input embed, shape is (bs, max_words, hid_dim)
        guidance_frame: (bs, frame, hid_dim)
        guidance_sentence: (bs, 1, hid_dim)
        attention_mask: (bs, max_words)
        prob = Tanh(MLP(x))
        '''      
        B, N, D = x.shape
        pred_score1 = self.score_predictor(x, guidance_frame).squeeze()  # (bs, max_words)
        pred_score2 = self.score_predictor(x, guidance_sentence).squeeze()  # (bs, max_words)
        #pred_score = pred_score1 * self.tanh(pred_score2)  # (bs, max_words)
        pred_score = pred_score1 + pred_score2  # (bs, max_words)

        # print("attention_mask: ", attention_mask[0], "\nattention_mask_new: ", attention_mask_new[0])
        #word_pred_score = pred_score*attention_mask # seperate the cls_token (bs, n_token)
        # print("word_pred_score: ", word_pred_score[0])
        topk_indicator = self.topk_selector(pred_score) # (bs, k, n_token))

        output = torch.einsum("bkl,bld->bkd", topk_indicator, x)

        return output
