import torch
import torch.nn as nn
from .attention import *
from .differential_topk import *

class GuidanceSeletectionBlock(nn.Module):
    def __init__(self, dim):
        super(GuidanceSeletectionBlock, self).__init__()
        self.trs_cat = nn.Linear(2*dim, dim)

    def forward(self, ego_feature, guidance_feature):
        # ego_faeture: [batch_size, n, dim] guidance_feature: [batch_size, 1, dim]
        guidance_feature = guidance_feature.repeat(1, ego_feature.size(1), 1)
        cat_features = torch.cat([ego_feature, guidance_feature], dim=2)
        weight = self.trs_cat(cat_features) * ego_feature
        weight = weight.mean(dim=2)
        weight = torch.softmax(weight, dim=1)
        weight = weight.unsqueeze(2)
        return weight*ego_feature


class FCDet(torch.nn.Module):
    def __init__(self,dataset, config):
        super(FCDet,self).__init__()
        self.num_frame = 20
        self.topk_token = config['topk_token']
        self.topk_patch = config['topk_patch']
        dropout = 0.3
        self.trs_sentence = nn.Sequential(nn.Linear(512,128),nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 128))
        self.trs_token = nn.Sequential(nn.Linear(512,128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 128))
        self.trs_frame = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 128))
        self.trs_patch = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 128))
        self.trs_reason = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 128))

        self.guidance_selection_patch = VisualTokenSelection(max_frames=self.num_frame,topk=self.topk_patch)
        self.guidance_selection_token = TextTokenSelection(topk=self.topk_token)
        self.selfatt_guidance_frame = SelfAttention(128)

        self.selfatt_token = SelfAttention(128)
        self.selfatt_patch = SelfAttention(128)

        self.co_attention_token_patch = CoSelection(128, 4, dropout, self.topk_token, self.num_frame*self.topk_patch)
        self.co_attention_sentence_frame = CoSelection(128, 4, dropout, 1, self.num_frame)
        self.transformer_token_patch = nn.TransformerEncoderLayer(d_model=128, nhead=4, batch_first=True)
        self.transformer_sentence_frame = nn.TransformerEncoderLayer(d_model=128, nhead=4, batch_first=True)
        self.transformer_coarse_reason = nn.TransformerEncoderLayer(d_model=128, nhead=4, batch_first=True)

        self.classifier_features = nn.Sequential(nn.Linear(128*2, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 2))
        self.classifier_reason = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 2))

        self.layer_norm = nn.LayerNorm(128, elementwise_affine=False)
        self.tanh = nn.Tanh()

    def cal_confidence(self, p):
        p = torch.softmax(p, dim=1)-0.5
        p = torch.abs(p)
        p = torch.mean(p, dim=1, keepdim=True)
        return p


    def forward(self, test=False, **kwargs):
        patch_features = kwargs['patch_features'] #128*20*16*512
        frame_features = kwargs['visual_features']  # 128*20*1*512
        token_features = kwargs['token_features']  # 128*77*512
        sentence_features = kwargs['text_features']  # 128*1*512
        reason_features = kwargs['reason_features']  # 128*77*512

        batch_size = patch_features.shape[0]

        patch_features = self.trs_patch(patch_features)
        frame_features = self.trs_frame(frame_features)
        token_features = self.trs_token(token_features)
        sentence_features = self.trs_sentence(sentence_features)

        ego_patch_features = patch_features.reshape(batch_size, -1, 128)
        guidance_frame_features1 = frame_features.squeeze(2)
        guidance_sentence_features1 = sentence_features
        update_patch_features = self.guidance_selection_patch(ego_patch_features, guidance_frame=guidance_frame_features1, guidance_sentence=guidance_sentence_features1)
        update_patch_features = self.layer_norm(update_patch_features) #128*20*k*128

        ego_token_features = token_features
        guidance_frame_features2 = self.selfatt_guidance_frame(frame_features.squeeze(2)).mean(1).unsqueeze(1)
        guidance_sentence_features2 = sentence_features
        update_token_features = self.guidance_selection_token(ego_token_features, guidance_frame=guidance_frame_features2, guidance_sentence=guidance_sentence_features2)
        update_token_features = self.layer_norm(update_token_features) #128*k*128

        co_token_features, co_patch_features = self.co_attention_token_patch(update_token_features, update_patch_features.reshape(batch_size,-1,128))
        fine_features = torch.cat([co_token_features, co_patch_features], dim=1)
        fine_features = self.transformer_token_patch(fine_features).mean(1)
        fine_features = self.layer_norm(fine_features)

        purify_frame_features = self.selfatt_patch(update_patch_features.reshape(batch_size*20,-1,128)).mean(1).reshape(batch_size,self.num_frame,128)
        purify_frame_features = self.layer_norm(purify_frame_features)
        purify_sentence_features = self.selfatt_token(update_token_features).mean(1).unsqueeze(1)
        purify_sentence_features = self.layer_norm(purify_sentence_features)

        co_sentence_features, co_frame_features = self.co_attention_sentence_frame(purify_sentence_features, purify_frame_features)
        coarse_features = torch.cat([co_sentence_features,co_frame_features],dim=1)
        coarse_features = self.transformer_sentence_frame(coarse_features)
        coarse_features = self.layer_norm(coarse_features)

        reason_features = self.trs_reason(reason_features)
        #reason_features = self.layer_norm(reason_features)
        reason_features = torch.cat([coarse_features, reason_features], dim=1)
        reason_features = self.transformer_coarse_reason(reason_features).mean(1)
        reason_features = self.layer_norm(reason_features)
        predict_reason = self.classifier_reason(reason_features)

        predict_features = self.classifier_features(torch.cat([fine_features, coarse_features.mean(1)], dim=1))

        output = self.cal_confidence(predict_features) * predict_features + self.cal_confidence(predict_reason) * predict_reason
        if test:
            return output, predict_reason, predict_features,self.gate_reason(predict_reason), self.gate_features(predict_features)
        return output, predict_reason, predict_features