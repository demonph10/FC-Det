import torch.nn as nn
from .trm import *
from einops import rearrange

class Attention(nn.Module):
    def __init__(self, dim, heads = 2, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class _MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, n_heads, dropout):
        super(_MultiHeadAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.n_heads = n_heads

        self.w_q = Linear(d_model, d_k * n_heads)
        self.w_k = Linear(d_model, d_k * n_heads)
        self.w_v = Linear(d_model, d_v * n_heads)

    def forward(self, q, k, v):
        # q: [b_size x len_q x d_model]
        # k: [b_size x len_k x d_model]
        # v: [b_size x len_k x d_model]
        b_size = q.size(0)

        # q_s: [b_size x n_heads x len_q x d_k]
        # k_s: [b_size x n_heads x len_k x d_k]
        # v_s: [b_size x n_heads x len_k x d_v]
        q_s = self.w_q(q).view(b_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.w_k(k).view(b_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.w_v(v).view(b_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        return q_s, k_s, v_s

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PoswiseFeedForwardNet, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNormalization(d_model)

    def forward(self, inputs):
        # inputs: [b_size x len_q x d_model]
        residual = inputs
        output = self.relu(self.conv1(inputs.transpose(1, 2)))

        # outputs: [b_size x len_q x d_model]
        output = self.conv2(output).transpose(1, 2)
        output = self.dropout(output)

        return self.layer_norm(residual + output)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_v, n_heads, dropout, d_model, visual_len, sen_len, fea_v, fea_s, pos):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.multihead_attn_v = _MultiHeadAttention(d_k, d_v, d_model, n_heads, dropout)
        self.multihead_attn_s = _MultiHeadAttention(d_k, d_v, d_model, n_heads, dropout)
        self.pos_emb_v = PosEncoding(visual_len * 10, d_model)
        self.pos_emb_s = PosEncoding(sen_len * 10, d_model)
        self.linear_v = nn.Linear(in_features=fea_v, out_features=d_model)
        self.linear_s = nn.Linear(in_features=fea_s, out_features=d_model)
        self.proj_v = Linear(n_heads * d_v, d_model)
        self.proj_s = Linear(n_heads * d_v, d_model)
        self.d_v = d_v
        self.dropout = nn.Dropout(dropout)
        self.layer_norm_v = LayerNormalization(d_model)
        self.layer_norm_s = LayerNormalization(d_model)
        self.attention = ScaledDotProductAttention(d_k, dropout)
        self.pos = pos

    def forward(self, v, s, v_len, s_len):
        b_size = v.size(0)
        # q: [b_size x len_q x d_model]
        # k: [b_size x len_k x d_model]
        # v: [b_size x len_v x d_model] note (len_k == len_v)
        v, s = self.linear_v(v), self.linear_s(s)
        if self.pos:
            pos_v, pos_s = self.pos_emb_v(v_len), self.pos_emb_s(s_len)
            residual_v, residual_s = v + pos_v, s + pos_s
        else:
            residual_v, residual_s = v, s
        # context: a tensor of shape [b_size x len_q x n_heads * d_v]
        q_v, k_v, v_v = self.multihead_attn_v(v, v, v)
        q_s, k_s, v_s = self.multihead_attn_s(s, s, s)
        context_v, attn_v = self.attention(q_v, k_s, v_s)
        context_s, attn_s = self.attention(q_s, k_v, v_v)
        context_v = context_v.transpose(1, 2).contiguous().view(b_size, -1, self.n_heads * self.d_v)
        context_s = context_s.transpose(1, 2).contiguous().view(b_size, -1, self.n_heads * self.d_v)
        # project back to the residual size, outputs: [b_size x len_q x d_model]
        output_v = self.dropout(self.proj_v(context_v))
        output_s = self.dropout(self.proj_s(context_s))
        return self.layer_norm_v(residual_v + output_v), self.layer_norm_s(residual_s + output_s)

class co_attention(nn.Module):
    def __init__(self, d_k, d_v, n_heads, dropout, d_model, visual_len, sen_len, fea_v, fea_s, pos):
        super(co_attention, self).__init__()
        self.multi_head = MultiHeadAttention(d_k=d_k, d_v=d_v, n_heads=n_heads, dropout=dropout, d_model=d_model,
                                             visual_len=visual_len, sen_len=sen_len, fea_v=fea_v, fea_s=fea_s, pos=pos)
        self.PoswiseFeedForwardNet_v = PoswiseFeedForwardNet(d_model=d_model, d_ff=128, dropout=dropout)
        self.PoswiseFeedForwardNet_s = PoswiseFeedForwardNet(d_model=d_model, d_ff=128,dropout=dropout)
    def forward(self, v, s, v_len, s_len):
        v, s = self.multi_head(v, s, v_len, s_len)
        v = self.PoswiseFeedForwardNet_v(v)
        s = self.PoswiseFeedForwardNet_s(s)
        return v, s


class CoSelection(nn.Module):
    def __init__(self, dim, n_heads, dropout, seq1, seq2):
        super(CoSelection, self).__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.multihead_attn_1 = _MultiHeadAttention(dim, dim, dim, n_heads, dropout)
        self.multihead_attn_2 = _MultiHeadAttention(dim, dim, dim, n_heads, dropout)
        self.attention_1 = ScaledDotProductAttention(dim, dropout)
        self.attention_2 = ScaledDotProductAttention(dim, dropout)
        self.fuse_k = nn.Linear(seq1+seq2, 1)
        self.fuse_v = nn.Linear(seq1+seq2, 1)
        self.proj_1 = Linear(n_heads * dim, dim)
        self.proj_2 = Linear(n_heads * dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNormalization(dim)

    def forward(self, feature1, feature2):
        # feature1: [b_size x len1 x d_model] feature2: [b_size x len2 x d_model]
        b_size = feature1.size(0)
        q_feature1, k_feature1, v_feature1 = self.multihead_attn_1(feature1, feature1, feature1)
        q_feature2, k_feature2, v_feature2 = self.multihead_attn_2(feature2, feature2, feature2)
        k_fuse = torch.cat([k_feature1, k_feature2], dim=2)
        k_fuse = self.fuse_k(k_fuse.transpose(2, 3)).transpose(2, 3)
        v_fuse = torch.cat([v_feature1, v_feature2], dim=2)
        v_fuse = self.fuse_v(v_fuse.transpose(2, 3)).transpose(2, 3)
        context_1, attn_1 = self.attention_1(q_feature1, k_fuse, v_fuse)
        context_2, attn_2 = self.attention_2(q_feature2, k_fuse, v_fuse)
        context_1 = context_1.transpose(1, 2).contiguous().view(b_size, -1, self.n_heads * self.dim)
        context_2 = context_2.transpose(1, 2).contiguous().view(b_size, -1, self.n_heads * self.dim)
        # project back to the residual size, outputs: [b_size x len_q x d_model]
        output_1 = self.dropout(self.proj_1(context_1))
        output_2 = self.dropout(self.proj_2(context_2))
        return self.layer_norm(feature1 + output_1), self.layer_norm(feature2 + output_2)

class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.scale = torch.sqrt(torch.FloatTensor([hidden_size])).cuda()

    def forward(self, inputs):
        # inputs: (batch_size, seq_length, hidden_size)

        Q = self.query(inputs)  # (batch_size, seq_length, hidden_size)
        K = self.key(inputs)  # (batch_size, seq_length, hidden_size)
        V = self.value(inputs)  # (batch_size, seq_length, hidden_size)

        scores = torch.matmul(Q, K.transpose(1, 2))  # (batch_size, seq_length, seq_length)
        scores = scores / self.scale

        attention_weights = torch.softmax(scores, dim=-1)  # (batch_size, seq_length, seq_length)

        weighted_sum = torch.matmul(attention_weights, V)  # (batch_size, seq_length, hidden_size)

        return weighted_sum
