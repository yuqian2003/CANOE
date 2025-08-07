import math
import torch
import os
from torch import nn
from module import (MultimodalContextualEmbedding as MyEmbedding, 
                   PositionalEncoding, 
                   SpatialTemporalPair as TransEncoder,
                   TemporalUserPair as ArrivalTime,
                   UserSpatialPair as UserNet,
                   CrossContextAttentiveDecoder as CrossAttention,
                   NextLocationPrediction as MyFullyConnect)
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda")
class MyModel_mp(nn.Module):
    def __init__(self, config, osc_type=1, bandwidth=None):
        super(MyModel_mp, self).__init__()
        self.config = config
        self.base_dim = config.Embedding.base_dim
        self.topic_num = config.Dataset.topic_num

        self.embedding_layer = MyEmbedding(config, bandwidth=bandwidth)


        self.fc_mapping = nn.Linear(config.Dataset.num_locations, 64)

        
        if config.Encoder.encoder_type == 'trans':
            emb_dim = self.base_dim
            self.positional_encoding = PositionalEncoding(emb_dim=emb_dim)
            self.encoder = TransEncoder(config)

        fc_input_dim = self.base_dim + self.base_dim

        if config.Model.at_type != 'none':
            self.at_net = ArrivalTime(config, osc_type=osc_type)
            fc_input_dim += self.base_dim

        if self.topic_num > 0:
            self.user_net = UserNet(input_dim=self.topic_num, output_dim=self.base_dim)
            fc_input_dim += self.base_dim

        self.mp1 = nn.Linear(40,64)
        
        self.mp2 = nn.Linear(self.base_dim,config.Dataset.num_locations)
        
        self.cross_attention = CrossAttention(query_dim=8, kv_dim=40, embed_dim=40, output_dim=config.Dataset.num_locations, num_heads=4, osc_type=osc_type)
        self.fc_layer = MyFullyConnect(input_dim=64,
                                       output_dim = config.Dataset.num_locations)
        self.rank_head = MyFullyConnect(input_dim=64,
                                       output_dim = config.Dataset.num_locations)
        
        self.out_dropout = nn.Dropout(0.1)

    def forward(self, batch_data):
        user_x = batch_data['user']
        loc_x = batch_data['location_x']
        hour_x = batch_data['hour']
        if self.topic_num > 0:
            pre_embedded = batch_data['user_topic_loc']
        batch_size, sequence_length = loc_x.shape

        loc_embedded, timeslot_embedded, smoothed_timeslot_embedded, user_embedded = self.embedding_layer(batch_data)
        time_embedded = timeslot_embedded[hour_x]

        lt_embedded = loc_embedded + time_embedded
        
        if self.config.Encoder.encoder_type == 'trans':
            future_mask = torch.triu(torch.ones(sequence_length, sequence_length), diagonal=1).to(lt_embedded.device)
            future_mask = future_mask.masked_fill(future_mask == 1, float('-inf')).bool()
            encoder_out = self.encoder(self.positional_encoding(lt_embedded * math.sqrt(self.base_dim)),
                                      src_mask=future_mask)
        
        combined = torch.cat([encoder_out, lt_embedded],dim = -1)
        encoder_out = encoder_out.view(-1, encoder_out.size(-1)) # useless 
        
        user_embedded = user_embedded[user_x]

        if self.config.Model.at_type != 'none':
            at_embedded, time_logits = self.at_net(timeslot_embedded,
                                      smoothed_timeslot_embedded,
                                      user_embedded,
                                      batch_data)
            combined = torch.cat([combined, at_embedded], dim=-1)

        combined_emb = torch.cat([combined, at_embedded], dim=-1)   # 时间 & sequential
        
        user_embedded = user_embedded.unsqueeze(1).repeat(1, sequence_length, 1)
        # a = self.at_proj(user_embedded)
        combined = torch.cat([combined, user_embedded], dim=-1)

        combined_emb_no_pre = torch.cat([combined_emb, user_embedded], dim=-1) # 时间 & sequential & 用户
        
        if self.topic_num > 0:
            pre_embedded = self.user_net(pre_embedded).unsqueeze(1).repeat(1, sequence_length, 1)
            combined = torch.cat([combined, pre_embedded], dim=-1)
        
 
        
        final_output = self.cross_attention(pre_embedded, combined_emb_no_pre, combined_emb_no_pre) 

        final_output = self.fc_mapping(final_output)

        combined = self.mp1(combined)
        residual_output = final_output + combined # [B, L, embed_dim] + [B, L, embed_dim]
        out = self.fc_layer(residual_output.view(batch_size * sequence_length, residual_output.shape[2]))
        rank = self.rank_head(residual_output.view(batch_size * sequence_length, residual_output.shape[2]))
        return out, time_logits ,rank
