import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
device = torch.device("cuda")

class Lee_Oscillator(nn.Module):
    def __init__(self, osc_type=1):
        super(Lee_Oscillator, self).__init__()
        self.osc_type = osc_type
        self.u = 0  # state variables of excitatory
        self.v = 0  # state variables of inhibitory
        self.uthreshold = 1
        self.vthreshold = 1
        
        # 根据表格设置参数
        if osc_type == 1:
            self.e1, self.e2, self.i1, self.i2, self.tau_e, self.tau_i, self.k = 0, 0, 5, 5, 1, 1, -500
        elif osc_type == 2:
            self.e1, self.e2, self.i1, self.i2, self.tau_e, self.tau_i, self.k = 1, 1, 5, 5, 5, -5, -500
        elif osc_type == 3:
            self.e1, self.e2, self.i1, self.i2, self.tau_e, self.tau_i, self.k = 0, 0, 5, 5, -5, -5, -500
        elif osc_type == 4:
            self.e1, self.e2, self.i1, self.i2, self.tau_e, self.tau_i, self.k = 1, 1, 5, 6, 1, 1, -500
        elif osc_type == 5:
            self.e1, self.e2, self.i1, self.i2, self.tau_e, self.tau_i, self.k = 1, 1, 5, 5, 5, 5, -500
        elif osc_type == 6:
            self.e1, self.e2, self.i1, self.i2, self.tau_e, self.tau_i, self.k = 0, 0, 5, 5, 5, -5, -500
        else:
            # 默认使用类型1的参数
            self.e1, self.e2, self.i1, self.i2, self.tau_e, self.tau_i, self.k = 0, 0, 5, 5, 1, 1, -500
        
        self.a1 = torch.tensor([self.i1]).to(device)
        self.a2 = torch.tensor([self.i2]).to(device)
        self.b1 = torch.tensor([self.e1]).to(device)
        self.b2 = torch.tensor([self.e2]).to(device)
        self.k = torch.tensor([self.k]).to(device)
        self.n = 50
    def Sig_pie(self, k):
        s = 13
        return (torch.exp(s * k) - torch.exp(-s * k)) / (torch.exp(s * k) + torch.exp(-s * k))
    def Calculatez(self, I):
        self.u = torch.randn(I.shape).to(device) * 0.01
        self.v = torch.randn(I.shape).to(device) * 0.01
        uv = torch.sub(self.u, self.v)
        kI = torch.mul(self.k, I)
        kI2 = torch.mul(kI, I)
        ###
        # kI2 = torch.clamp(kI2, min=-50, max=50)
        z = torch.add(torch.mul(uv, torch.exp(kI2)), F.relu(I))
        # z = torch.add(torch.mul(uv, torch.exp(kI2)), self.Sig_pie(I))
        return z
    def forward(self, I):
        self.u = torch.zeros(I.shape).to(device)
        self.v = torch.zeros(I.shape).to(device)
        self.uthreshold = torch.tensor(0).to(device)
        self.vthreshold = torch.tensor(0).to(device)  # 节省内存，不zeros(I.shape)
        z = self.Calculatez(I)

        for i in range(self.n):
            self.u = F.relu(torch.add(torch.add(torch.mul(self.a1, self.u), torch.mul(self.a2, self.v)),
                                      torch.sub(I, self.uthreshold)))
            self.v = F.relu(torch.sub(torch.sub(torch.mul(self.b1, self.u), torch.mul(self.b2, self.v)), self.vthreshold))
        z = self.Calculatez(I)  # z只需要在最后计算一次

        return z


# Multimodal Contextual Embedding Module
class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.emb_dim = emb_dim
        pos_encoding = torch.zeros(max_len, emb_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * -(math.log(10000.0) / emb_dim))
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        pos_encoding = pos_encoding.unsqueeze(0)
        self.register_buffer("pos_encoding", pos_encoding)
        self.dropout = nn.Dropout(0.1)

    def forward(self, out):
        out = out + self.pos_encoding[:, :out.size(1)].detach()
        out = self.dropout(out)
        return out

class MultimodalContextualEmbedding(nn.Module):
    def __init__(self, config, bandwidth=None):
        super(MultimodalContextualEmbedding, self).__init__()
        self.config = config
        self.num_locations = config.Dataset.num_locations
        self.base_dim = config.Embedding.base_dim
        self.num_users = config.Dataset.num_users
        self.user_embedding = nn.Embedding(self.num_users, self.base_dim)
        self.location_embedding = nn.Embedding(self.num_locations, self.base_dim)
        self.timeslot_embedding = nn.Embedding(24, self.base_dim)
        
        # 如果提供了bandwidth参数，使用它；否则使用默认值
        if bandwidth is not None:
            self.bandwidth = nn.Parameter(torch.tensor([bandwidth], dtype=torch.float))
        else:
            self.bandwidth = nn.Parameter(torch.ones(1) + 1e-6)
            
        self.max_seq_length = 64
    
    def gaussian_kernel(self, timestamps, tn):
        dist = torch.min(torch.abs(timestamps - tn), 24 - torch.abs(timestamps - tn))
        return torch.exp(-0.5 * (dist / self.bandwidth) ** 2)
    
    def forward(self, batch_data):
        location_x = batch_data['location_x']
        loc_embedded = self.location_embedding(location_x)
        user_embedded = self.user_embedding(torch.arange(end=self.num_users, dtype=torch.int, device=location_x.device))
        timeslot_embedded = self.timeslot_embedding(torch.arange(end=24, dtype=torch.int, device=location_x.device))
        
        smoothed_list = []
        for tn in range(24):
            kernel_weights = self.gaussian_kernel(torch.arange(24, device=location_x.device), tn).view(24, 1)
            smoothed = torch.sum(kernel_weights * timeslot_embedded, dim=0)
            smoothed_list.append(smoothed)
        
        smoothed_timeslot_embedded = torch.stack(smoothed_list, dim=0)
        return loc_embedded, timeslot_embedded, smoothed_timeslot_embedded, user_embedded

# Tri-Pair Interaction Encoder
class UserSpatialPair(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(UserSpatialPair, self).__init__()
        self.topic_num = input_dim
        self.output_dim = output_dim
        self.block = nn.Sequential(
            nn.Linear(self.topic_num, self.topic_num * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.topic_num * 2, self.topic_num)
        )
        self.final = nn.Sequential(
            nn.LayerNorm(self.topic_num),
            nn.Linear(self.topic_num, self.output_dim)
        )

    def forward(self, topic_vec):
        x = topic_vec
        topic_vec = self.block(topic_vec)
        topic_vec = x + topic_vec
        return self.final(topic_vec)

class TemporalUserPair(nn.Module):
    def __init__(self, config, osc_type=1):
        super(TemporalUserPair, self).__init__()
        self.config = config
        self.base_dim = config.Embedding.base_dim
        self.num_heads = 4
        assert self.base_dim % self.num_heads == 0, "base_dim must be divisible by num_heads"
        self.head_dim = self.base_dim // self.num_heads
        self.num_users = config.Dataset.num_users
        self.timeslot_num = 24
        
        if config.Model.at_type == 'attn':
            self.user_preference = nn.Embedding(self.num_users, self.base_dim)
            self.w_q = nn.ModuleList(
                 [nn.Linear(self.base_dim * 2, self.head_dim) for _ in range(self.num_heads)])
            self.w_k = nn.ModuleList(
                [nn.Linear(self.base_dim, self.head_dim) for _ in range(self.num_heads)])
            self.w_v = nn.ModuleList(
                [nn.Linear(self.base_dim, self.head_dim) for _ in range(self.num_heads)])
            self.unify_heads = nn.Linear(self.base_dim, self.base_dim)
            self.key_trans = nn.Linear(in_features=5120, out_features=24, bias=True)
            self.query_trans = nn.Linear(in_features=24, out_features=5120, bias=True)
        self.Lee_Oscillator = Lee_Oscillator(osc_type=osc_type)
        self.time_head = nn.Linear(self.base_dim, self.timeslot_num)

    def forward(self, timeslot_embedded, smoothed_timeslot_embedded, user_embedded, batch_data):
        user_x = batch_data['user']
        hour_x = batch_data['hour']
        batch_size, sequence_length = hour_x.shape
        total_sequences = batch_size * sequence_length
        hour_mask = batch_data['hour_mask'].view(batch_size * sequence_length, -1)
        
        if self.config.Model.at_type == 'attn':
            hour_x = hour_x.view(batch_size * sequence_length)
            head_outputs = []
            user_preference = self.user_preference(user_x).unsqueeze(1).repeat(1, sequence_length, 1)
            user_feature = user_preference.view(batch_size * sequence_length, -1)
            time_feature = timeslot_embedded[hour_x]
            
            query = torch.cat([user_feature, time_feature], dim=-1)
            key = timeslot_embedded
            for i in range(self.num_heads):
                query_i = self.w_q[i](query)
                key_i = self.w_k[i](key)
                value_i = self.w_v[i](key)
                attn_scores_i = torch.matmul(query_i, key_i.T)
                scale = 1.0 / (key_i.size(-1) ** 0.5)
                attn_scores_i = attn_scores_i * scale
                attn_scores_i = attn_scores_i.masked_fill(hour_mask == 1, float('-inf'))
                attn_scores_i = torch.softmax(attn_scores_i, dim=-1)
                weighted_values_i = torch.matmul(attn_scores_i, value_i)
                head_outputs.append(weighted_values_i)
            head_outputs = torch.cat(head_outputs, dim=-1)
            head_outputs = head_outputs.view(batch_size, sequence_length, -1)
            at_emb = self.unify_heads(head_outputs)
            time_logits = self.time_head(head_outputs)
        return at_emb, time_logits

class SpatialTemporalPair(nn.Module):
    def __init__(self, config):
        super(SpatialTemporalPair, self).__init__()
        self.config = config
        input_dim = config.Embedding.base_dim

        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim,
                                                   activation='gelu',
                                                   batch_first=True,
                                                   dim_feedforward=input_dim,
                                                   nhead=4,
                                                   dropout=0.1)

        encoder_norm = nn.LayerNorm(input_dim)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer,
                                             num_layers=3,
                                             norm=encoder_norm)
        self.initialize_parameters()

    def forward(self, embedded_out, src_mask):
        out = self.encoder(embedded_out, mask=src_mask)
        return out

    def initialize_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                init.xavier_uniform_(p)

# Cross Context Attentive Decoder
class CrossContextAttentiveDecoder(nn.Module):
    def __init__(self, query_dim, kv_dim, embed_dim, output_dim, num_heads=4, osc_type=1):
        super(CrossContextAttentiveDecoder, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        
        self.query_proj = nn.Linear(query_dim, embed_dim)
        self.k_proj = nn.Linear(kv_dim, embed_dim) if kv_dim != embed_dim else nn.Identity()
        self.v_proj = nn.Linear(kv_dim, embed_dim) if kv_dim != embed_dim else nn.Identity()
        
        self.Lee_Oscillator = Lee_Oscillator(osc_type=osc_type)
        self.out_fc = nn.Linear(embed_dim, output_dim)

    def forward(self, query, key, value):
        B, L_q, _ = query.size()
        B, L_k, _ = key.size()
        Q = self.query_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        Q = Q.view(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, L_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L_k, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores_flat = scores.contiguous().view(B * self.num_heads, L_q, L_k)
        scores = scores_flat.view(B, self.num_heads, L_q, L_k)
        attn_weights = self.Lee_Oscillator(scores)
        attn_weights = torch.softmax(scores, dim=-1)
        head_outputs = torch.matmul(attn_weights, V)
        head_outputs = head_outputs.transpose(1, 2).contiguous().view(B, L_q, self.embed_dim)

        out = self.out_fc(head_outputs)
        return out

# Next Location Prediction
class NextLocationPrediction(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NextLocationPrediction, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim*2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim*2, input_dim),
            nn.Dropout(0.1),
        )

        self.batch_norm = nn.BatchNorm1d(input_dim)
        self.drop = nn.Dropout(0.1)

        num_locations = output_dim
        self.linear_class1 = nn.Linear(input_dim, num_locations)

    def forward(self, out):
        x = out
        out = self.block(out)
        out = out + x
        out = self.batch_norm(out)
        out = self.drop(out)
        return self.linear_class1(out) 