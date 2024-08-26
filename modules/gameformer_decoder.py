import torch

import math
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, max_len=100):
        super(PositionalEncoding, self).__init__()
        d_model = 256
        dropout = 0.1
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.permute(1, 0, 2)
        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x + self.pe
        
        return self.dropout(x)
    

class AgentEncoder(nn.Module):
    def __init__(self):
        super(AgentEncoder, self).__init__()
        self.motion = nn.LSTM(8, 256, 2, batch_first=True)
        self.type_emb = nn.Embedding(4, 256, padding_idx=0)

    def forward(self, inputs):
        traj, _ = self.motion(inputs[:, :, :8])
        output = traj[:, -1]
        type = self.type_emb(inputs[:, -1, 8].int())
        output = output + type

        return output
    

class LaneEncoder(nn.Module):
    def __init__(self):
        super(LaneEncoder, self).__init__()
        # encdoer layer
        self.self_line = nn.Linear(3, 128)
        self.left_line = nn.Linear(3, 128)
        self.right_line = nn.Linear(3, 128)
        self.speed_limit = nn.Linear(1, 64)
        self.self_type = nn.Embedding(4, 64, padding_idx=0)
        self.left_type = nn.Embedding(11, 64, padding_idx=0)
        self.right_type = nn.Embedding(11, 64, padding_idx=0)
        self.traffic_light_type = nn.Embedding(9, 64, padding_idx=0)
        self.interpolating = nn.Embedding(2, 64)
        self.stop_sign = nn.Embedding(2, 64)

        # hidden layers
        self.pointnet = nn.Sequential(nn.Linear(512, 384), nn.ReLU(), nn.Linear(384, 256))
        self.position_encode = PositionalEncoding(max_len=100)

    def forward(self, inputs):
        # embedding
        self_line = self.self_line(inputs[..., :3])
        left_line = self.left_line(inputs[..., 3:6])
        right_line = self.right_line(inputs[...,  6:9])
        speed_limit = self.speed_limit(inputs[..., 9].unsqueeze(-1))
        self_type = self.self_type(inputs[..., 10].int())
        left_type = self.left_type(inputs[..., 11].int())
        right_type = self.right_type(inputs[..., 12].int()) 
        traffic_light = self.traffic_light_type(inputs[..., 13].int())
        interpolating = self.interpolating(inputs[..., 14].int()) 
        stop_sign = self.stop_sign(inputs[..., 15].int())

        lane_attr = self_type + left_type + right_type + traffic_light + interpolating + stop_sign
        lane_embedding = torch.cat([self_line, left_line, right_line, speed_limit, lane_attr], dim=-1)
    
        # process
        output = self.position_encode(self.pointnet(lane_embedding))

        return output
    

class CrosswalkEncoder(nn.Module):
    def __init__(self):
        super(CrosswalkEncoder, self).__init__()
        self.point_net = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 256))
    
    def forward(self, inputs):
        output = self.point_net(inputs)

        return output
    

class FutureEncoder(nn.Module):
    def __init__(self):
        super(FutureEncoder, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 256))
        self.type_emb = nn.Embedding(4, 256, padding_idx=0)

    def state_process(self, trajs, current_states):
        M = trajs.shape[2]
        current_states = current_states.unsqueeze(2).expand(-1, -1, M, -1)
        xy = torch.cat([current_states[:, :, :, None, :2], trajs], dim=-2)
        dxy = torch.diff(xy, dim=-2)
        v = dxy / 0.1
        theta = torch.atan2(dxy[..., 1], dxy[..., 0].clamp(min=1e-3)).unsqueeze(-1)
        T = trajs.shape[3]
        size = current_states[:, :, :, None, 5:8].expand(-1, -1, -1, T, -1)
        trajs = torch.cat([trajs, theta, v, size], dim=-1) # (x, y, heading, vx, vy, w, l, h)

        return trajs

    def forward(self, current_states):
        # trajs = self.state_process(trajs, current_states)
        trajs = self.mlp(current_states.detach())
        # type = self.type_emb(current_states[:, :, None, 8].int())
        output = torch.max(trajs, dim=-2).values
        # output = output + type

        return output


class GMMPredictor(nn.Module):
    def __init__(self, future_len):
        super(GMMPredictor, self).__init__()
        self._future_len = future_len
        self.gaussian = nn.Sequential(nn.Linear(256, 512), nn.ELU(), nn.Dropout(0.1), nn.Linear(512, self._future_len*4))
        self.score = nn.Sequential(nn.Linear(256, 64), nn.ELU(), nn.Dropout(0.1), nn.Linear(64, 1))
    
    def forward(self, input):
        res = self.gaussian(input).view(*input.shape[:-1],self._future_len, 4)
        score = self.score(input).squeeze(-1)

        return res, score


class SelfTransformer(nn.Module):
    def __init__(self):
        super(SelfTransformer, self).__init__()
        heads, dim, dropout = 8, 256, 0.1
        self.self_attention = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim*4, dim), nn.Dropout(dropout))

    def forward(self, inputs, mask=None):
        attention_output, _ = self.self_attention(inputs, inputs, inputs, key_padding_mask=mask)
        attention_output = self.norm_1(attention_output + inputs)
        output = self.norm_2(self.ffn(attention_output) + attention_output)

        return output


class CrossTransformer(nn.Module):
    def __init__(self):
        super(CrossTransformer, self).__init__()
        heads, dim, dropout = 8, 256, 0.1
        self.cross_attention = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim*4, dim), nn.Dropout(dropout))

    def forward(self, query, key, value, mask=None):
        attention_output, _ = self.cross_attention(query, key, value, key_padding_mask=mask)
        attention_output = self.norm_1(attention_output)
        output = self.norm_2(self.ffn(attention_output) + attention_output)

        return output


class InitialDecoder(nn.Module):
    def __init__(self, modalities, neighbors, future_len):
        super(InitialDecoder, self).__init__()
        dim = 256
        self._modalities = modalities
        self.multi_modal_query_embedding = nn.Embedding(modalities, dim)
        self.agent_query_embedding = nn.Embedding(neighbors+1, dim)
        self.query_encoder = CrossTransformer()
        self.predictor = GMMPredictor(future_len)
        self.register_buffer('modal', torch.arange(modalities).long())
        self.register_buffer('agent', torch.arange(neighbors+1).long())

    def forward(self, id, encoding, mask):
        # get query
        B,_,dim=encoding.shape
        multi_modal_query = self.multi_modal_query_embedding(self.modal)
        agent_query = self.agent_query_embedding(self.agent[id])
        multi_modal_agent_query = multi_modal_query + agent_query[None, :]
        query = ((encoding[:,:,None,:]+multi_modal_agent_query)).reshape(B,-1,dim)

        # decode trajectories
        query_content = self.query_encoder(query, encoding, encoding, mask)
        return query_content.reshape(B,-1,self._modalities,dim)


class InteractionDecoder(nn.Module):
    def __init__(self, future_encoder, future_len):
        super(InteractionDecoder, self).__init__()
        self.interaction_encoder = SelfTransformer()
        self.query_encoder = CrossTransformer()
        self.future_encoder = future_encoder
        self.decoder = GMMPredictor(future_len)

    def forward(self, id, actors, scores, last_content, encoding, mask):
        B, N, M, T, _ = actors.shape
        
        # encoding the trajectories from the last level 
        multi_futures = self.future_encoder(actors[..., :2])
        futures = (multi_futures * scores.softmax(-1).unsqueeze(-1)).mean(dim=2)

        # encoding the interaction using self-attention transformer   
        interaction = self.interaction_encoder(futures, mask)

        # append the interaction encoding to the context encoding
        encoding = torch.cat([interaction, encoding], dim=1)
        mask = torch.cat([mask[:, :N], mask], dim=1).clone()
        mask[:, id] = True # mask the agent future itself from last level

        # decoding the trajectories from the current level
        query = last_content + multi_futures
        query_content = self.query_encoder(query.reshape(B,-1,query.shape[-1]), encoding, encoding, mask)
        query_content=query_content.reshape(B,N,-1,query.shape[-1])
        trajectories, scores = self.decoder(query_content)

        # post process

        return query_content, trajectories, scores
class Decoder(nn.Module):
    def __init__(self, modalities, future_len, neighbors_to_predict, levels=3):
        super(Decoder, self).__init__()
        self._levels = levels
        self._neighbors = neighbors_to_predict
        future_encoder = FutureEncoder()
        self.initial_stage = InitialDecoder(modalities, neighbors_to_predict, future_len)
        self.interaction_stage = nn.ModuleList([InteractionDecoder(future_encoder, future_len) for _ in range(levels)])  
        self.embedmap=nn.Linear(6*128,256)

    def forward(self, encoder_inputs):
        decoder_outputs = {}
        # assert encoder_inputs['actors'].shape[1] >= N, 'Too many neighbors to predict'
        last_level=torch.nested.nested_tensor(list(torch.tensor_split(encoder_inputs["pred"]['loc_propose_pos'],encoder_inputs['ptr'][:-1]))).to_padded_tensor(0)
        last_scores=torch.nested.nested_tensor(list(torch.tensor_split(encoder_inputs["pred"]['pi'],encoder_inputs['ptr'][:-1]))).to_padded_tensor(0)
        # current_states = encoder_inputs['actors'][:, :, -1]
        input=torch.nested.nested_tensor(list(torch.tensor_split( self.embedmap(encoder_inputs['encodings']),encoder_inputs['ptr'][:-1])))
        encodings=input.to_padded_tensor(0)
        masks=torch.ones_like(input).to_padded_tensor(0).all(-1)
        masks=~masks
        last_content = self.initial_stage(0, encodings, masks)
        decoder_outputs['level_0_interactions'] = last_level
        decoder_outputs['level_0_scores'] = last_scores
        # level k reasoning
        for k in range(1, self._levels+1):
            interaction_decoder = self.interaction_stage[k-1]
            results = interaction_decoder(0, last_level, last_scores,last_content, encodings, masks)
            last_content = results[0]
            last_level = results[1]
            last_scores = results[2]
            decoder_outputs[f'level_{k}_interactions'] = last_level
            decoder_outputs[f'level_{k}_scores'] = last_scores

        return decoder_outputs