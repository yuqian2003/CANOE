import math
import torch
from torch import nn
device = torch.device("cuda")

class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.emb_dim = emb_dim
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, emb_dim))
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

class MyEmbedding(nn.Module):
    def __init__(self, args, num_locations, num_users):
        super(MyEmbedding, self).__init__()
        self.num_locations = num_locations
        self.base_dim = args.dim
        self.num_users = num_users

        self.user_embedding = nn.Embedding(self.num_users, self.base_dim)
        self.location_embedding = nn.Embedding(self.num_locations, self.base_dim)
        self.timeslot_embedding = nn.Embedding(24, self.base_dim)

        # self.bandwidth = nn.Parameter(torch.ones(1) + 1e-6)
        self.bandwidth = args.bandwidth
        self.max_seq_length = 64  
    
    def gaussian_kernel(self, timestamps, tn):
        # dist = torch.abs(timestamps - tn)  -- considering periodicity
        dist = torch.min(torch.abs(timestamps - tn), 24 - torch.abs(timestamps - tn))
        return torch.exp(-0.5 * (dist / self.bandwidth) ** 2)

    
    def forward(self, batch_data):
        location_x = batch_data['location_x']
        
        loc_embedded = self.location_embedding(location_x)
        user_embedded = self.user_embedding(torch.arange(end=self.num_users, dtype=torch.int, device=location_x.device))
        timeslot_embedded = self.timeslot_embedding(torch.arange(end=24, dtype=torch.int, device=location_x.device))  # Shape: [24, base_dim]
        
        #smoothed_timeslot_embedded = []
        smoothed_list = []
        for tn in range(24):
            kernel_weights = self.gaussian_kernel(torch.arange(24, device=location_x.device), tn).view(24, 1)
            smoothed = torch.sum(kernel_weights * timeslot_embedded, dim=0)  # Shape: [base_dim]
            smoothed_list.append(smoothed)
        
        smoothed_timeslot_embedded = torch.stack(smoothed_list, dim=0)  # [24, 16]
        
        return loc_embedded, timeslot_embedded, smoothed_timeslot_embedded, user_embedded



