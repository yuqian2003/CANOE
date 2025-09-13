import os
import math
import torch
from torch import nn
from cnolp_module import MultimodalContextualEmbedding, PositionalEncoding, UserLocationPair, TimeUserPair, NextLocationPrediction, CrossContextAttentiveDecoder, LocationTimePair
device = torch.device("cuda")

class CNOLP_tc(nn.Module):
    def __init__(self, args, num_locations, num_users, topic_num):
        super(CNOLP_tc, self).__init__()
        self.base_dim = args.dim
        self.topic_num = topic_num
        self.num = num_locations
        self.embedding_layer = MultimodalContextualEmbedding(args, num_locations, num_users)

        self.fc_mapping = nn.Linear(num_locations, 80)
        if args.encoder == 'trans':
            emb_dim = self.base_dim
            self.positional_encoding = PositionalEncoding(emb_dim=emb_dim)
            self.encoder = LocationTimePair(args)

        fc_input_dim = self.base_dim + self.base_dim

        if args.at != 'none':
            self.at_net = TimeUserPair(args, num_users)
            fc_input_dim += self.base_dim

        if self.topic_num > 0:
            self.user_net = UserLocationPair(input_dim=self.topic_num, output_dim=self.base_dim)
            fc_input_dim += self.base_dim
        self.cnoa = CrossContextAttentiveDecoder(query_dim=16, kv_dim=80, embed_dim=80, output_dim=num_locations, args=args, num_heads=4)
        self.fc_layer = NextLocationPrediction(input_dim = self.base_dim * 5,
                                       output_dim = num_locations)
        self.rank_head = NextLocationPrediction(input_dim = self.base_dim * 5,
                                       output_dim = num_locations)
        self.out_dropout = nn.Dropout(0.1)
        
    def forward(self, batch_data, args):
        user_x = batch_data['user']
        loc_x = batch_data['location_x']
        hour_x = batch_data['hour']
        if self.topic_num > 0:
            pre_embedded = batch_data['user_topic_loc']
        batch_size, sequence_length = loc_x.shape
        loc_embedded, timeslot_embedded, smoothed_timeslot_embedded, user_embedded = self.embedding_layer(batch_data)

        
        time_embedded = timeslot_embedded[hour_x]
        smoothed_time_embedded = smoothed_timeslot_embedded[hour_x]

        lt_embedded = loc_embedded + time_embedded
        if hasattr(self, 'encoder'):
            future_mask = torch.triu(torch.ones(sequence_length, sequence_length), diagonal=1).to(lt_embedded.device)
            future_mask = future_mask.masked_fill(future_mask == 1, float('-inf')).bool()
            encoder_out = self.encoder(self.positional_encoding(lt_embedded * math.sqrt(self.base_dim)),
                                      src_mask=future_mask)

        combined = torch.cat([encoder_out,lt_embedded],dim = -1)
        encoder_out = encoder_out.view(-1, encoder_out.size(-1)) 
        
        user_embedded = user_embedded[user_x]

        if hasattr(self, 'at_net'):
            at_embedded, time_logits = self.at_net(timeslot_embedded,
                                      smoothed_timeslot_embedded,
                                      user_embedded,
                                      batch_data
                                      )
            
            combined = torch.cat([combined, at_embedded], dim=-1)
            combined_emb = torch.cat([combined, at_embedded], dim=-1)  
        
        user_embedded = user_embedded.unsqueeze(1).repeat(1, sequence_length, 1)
        
        combined = torch.cat([combined, user_embedded], dim=-1)

        combined_emb_no_pre = torch.cat([combined_emb, user_embedded], dim=-1) # time & sequential & user
        
        if self.topic_num > 0:
            pre_embedded = self.user_net(pre_embedded).unsqueeze(1).repeat(1, sequence_length, 1)
            combined = torch.cat([combined, pre_embedded], dim=-1)
            
        final_output = self.cnoa(pre_embedded, combined_emb_no_pre, combined_emb_no_pre)

        final_output = self.fc_mapping(final_output)
        residual_output = final_output + combined
        out = self.fc_layer(residual_output.view(batch_size * sequence_length, residual_output.shape[2]))
        rank = self.rank_head(residual_output.view(batch_size * sequence_length, residual_output.shape[2]))  
        return out, time_logits ,rank

class CNOLP_mp(nn.Module):
    def __init__(self, args, num_locations, num_users, topic_num):
        super(CNOLP_mp, self).__init__()
        self.base_dim = args.dim
        self.topic_num = topic_num
        self.embedding_layer = MultimodalContextualEmbedding(args, num_locations, num_users)
        self.fc_mapping = nn.Linear(num_locations, 64)
        if args.encoder == 'trans':
            emb_dim = self.base_dim
            self.positional_encoding = PositionalEncoding(emb_dim=emb_dim)
            self.encoder = LocationTimePair(args)
        fc_input_dim = self.base_dim + self.base_dim
        if args.at != 'none':
            self.at_net =  TimeUserPair(args, num_users)
            fc_input_dim += self.base_dim
        if self.topic_num > 0: # LDA
            self.user_net = UserLocationPair(input_dim=self.topic_num, output_dim=self.base_dim)
            fc_input_dim += self.base_dim
        self.mp1 = nn.Linear(40,64)
        #self.cnoa = CrossContextAttentiveDecoder(query_dim=8, 
        #                                                    kv_dim=40, 
        #                                                    embed_dim=40, 
        #                                                    output_dim=num_locations,
        #                                                    num_heads=4)
        self.cnoa = CrossContextAttentiveDecoder(query_dim=8, kv_dim=40, embed_dim=40, output_dim=num_locations, args=args, num_heads=4)
        self.fc_layer = NextLocationPrediction(input_dim=64,
                                       output_dim = num_locations)
        self.rank_head = NextLocationPrediction(input_dim=64,
                                       output_dim = num_locations)
        self.out_dropout = nn.Dropout(0.1)

    def forward(self, batch_data, args):
        user_x = batch_data['user']
        loc_x = batch_data['location_x']
        hour_x = batch_data['hour']
        if self.topic_num > 0:
            pre_embedded = batch_data['user_topic_loc']
        batch_size, sequence_length = loc_x.shape

        loc_embedded, timeslot_embedded, smoothed_timeslot_embedded, user_embedded = self.embedding_layer(batch_data)
        time_embedded = timeslot_embedded[hour_x]

        lt_embedded = loc_embedded + time_embedded
        
        if args.encoder == 'trans':
            future_mask = torch.triu(torch.ones(sequence_length, sequence_length), diagonal=1).to(lt_embedded.device)
            future_mask = future_mask.masked_fill(future_mask == 1, float('-inf')).bool()
            encoder_out = self.encoder(self.positional_encoding(lt_embedded * math.sqrt(self.base_dim)),
                                      src_mask=future_mask)
        
        combined = torch.cat([encoder_out, lt_embedded],dim = -1)
        encoder_out = encoder_out.view(-1, encoder_out.size(-1)) 
        
        user_embedded = user_embedded[user_x]

        if args.at != 'none':
            at_embedded, time_logits = self.at_net(timeslot_embedded,
                                      smoothed_timeslot_embedded,
                                      user_embedded,
                                      batch_data)
            combined = torch.cat([combined, at_embedded], dim=-1)

        combined_emb = torch.cat([combined, at_embedded], dim=-1)   # time & sequential
        
        user_embedded = user_embedded.unsqueeze(1).repeat(1, sequence_length, 1)
        combined = torch.cat([combined, user_embedded], dim=-1)

        combined_emb_no_pre = torch.cat([combined_emb, user_embedded], dim=-1) # time & sequential & user
        
        if self.topic_num > 0:
            pre_embedded = self.user_net(pre_embedded).unsqueeze(1).repeat(1, sequence_length, 1)
            combined = torch.cat([combined, pre_embedded], dim=-1)
        
        final_output = self.cnoa(pre_embedded, combined_emb_no_pre, combined_emb_no_pre) 

        final_output = self.fc_mapping(final_output)

        combined = self.mp1(combined)
        residual_output = final_output + combined # [B, L, embed_dim] + [B, L, embed_dim]
        out = self.fc_layer(residual_output.view(batch_size * sequence_length, residual_output.shape[2]))
        rank = self.rank_head(residual_output.view(batch_size * sequence_length, residual_output.shape[2]))
        return out, time_logits ,rank
