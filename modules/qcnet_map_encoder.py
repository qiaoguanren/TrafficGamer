from typing import Dict

import torch
import torch.nn as nn
from torch_cluster import radius_graph
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData

from layers.attention_layer import AttentionLayer
from layers.fourier_embedding import FourierEmbedding
from utils import angle_between_2d_vectors
from utils import merge_edges
from utils import weight_init
from utils import wrap_angle


class QCNetMapEncoder(nn.Module):

    def __init__(self,
                 dataset: str,
                 input_dim: int,
                 hidden_dim: int,
                 num_historical_steps: int,
                 pl2pl_radius: float,
                 num_freq_bands: int,
                 num_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float) -> None:
        super(QCNetMapEncoder, self).__init__()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.pl2pl_radius = pl2pl_radius
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout

        if dataset == 'argoverse_v2':
            if input_dim == 2:
                input_dim_x_pt = 1
                input_dim_x_pl = 0
                input_dim_stop_sign_pl = 0
                input_dim_tfc_sig_pl = 0
                input_dim_r_pt2pl = 3
                input_dim_r_pl2pl = 3
            elif input_dim == 3:
                input_dim_x_pt = 2
                input_dim_x_pl = 0
                input_dim_stop_sign_pl = 0
                input_dim_tfc_sig_pl = 0
                input_dim_r_pt2pl = 4
                input_dim_r_pl2pl = 4
            else:
                raise ValueError('{} is not a valid dimension'.format(input_dim))
        elif dataset == 'waymo':
            if input_dim == 2:
                input_dim_x_pt = 1
                input_dim_x_pl = 1
                input_dim_stop_sign_pl = 2
                input_dim_tfc_sig_pl = 2
                input_dim_r_pt2pl = 3
                input_dim_r_pl2pl = 3
            elif input_dim == 3:
                input_dim_x_pt = 2
                input_dim_x_pl = 1
                input_dim_stop_sign_pl = 3
                input_dim_tfc_sig_pl = 3
                input_dim_r_pt2pl = 4
                input_dim_r_pl2pl = 4
            else:
                raise ValueError('{} is not a valid dimension'.format(input_dim))
        else:
            raise ValueError('{} is not a valid dataset'.format(dataset))

        if dataset == 'argoverse_v2':
            self.type_pt_emb = nn.Embedding(17, hidden_dim)
            self.side_pt_emb = nn.Embedding(3, hidden_dim)
            self.type_pl_emb = nn.Embedding(4, hidden_dim)
            self.tfc_pl_emb = None
            self.turn_pl_emb = None
            self.int_pl_emb = nn.Embedding(3, hidden_dim)
            self.no_stop_sign_pl_emb = None
            self.tfc_sig_state_pl_emb = None
        elif dataset == 'waymo':
            self.type_pt_emb = nn.Embedding(16, hidden_dim)
            self.side_pt_emb = nn.Embedding(3, hidden_dim)
            self.type_pl_emb = nn.Embedding(7, hidden_dim)
            self.tfc_pl_emb = None
            self.turn_pl_emb = None
            self.int_pl_emb = None
            self.no_stop_sign_pl_emb = nn.Embedding(1, hidden_dim)
            self.tfc_sig_state_pl_emb = nn.Embedding(10, hidden_dim)
        else:
            raise ValueError('{} is not a valid dataset'.format(dataset))
        self.type_pl2pl_emb = nn.Embedding(5, hidden_dim)
        self.x_pt_emb = FourierEmbedding(input_dim=input_dim_x_pt, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.x_pl_emb = FourierEmbedding(input_dim=input_dim_x_pl, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        if dataset == 'waymo':
            self.stop_sign_pl_emb = FourierEmbedding(input_dim=input_dim_stop_sign_pl, hidden_dim=hidden_dim,
                                                     num_freq_bands=num_freq_bands)
            self.tfc_sig_pl_emb = FourierEmbedding(input_dim=input_dim_tfc_sig_pl, hidden_dim=hidden_dim,
                                                   num_freq_bands=num_freq_bands)
        else:
            self.stop_sign_pl_emb = None
            self.tfc_sig_pl_emb = None
        self.r_pt2pl_emb = FourierEmbedding(input_dim=input_dim_r_pt2pl, hidden_dim=hidden_dim,
                                            num_freq_bands=num_freq_bands)
        self.r_pl2pl_emb = FourierEmbedding(input_dim=input_dim_r_pl2pl, hidden_dim=hidden_dim,
                                            num_freq_bands=num_freq_bands)
        self.pt2pl_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.pl2pl_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=False, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.apply(weight_init)

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        pos_pt = data['map_point']['position'][:, :self.input_dim].contiguous()
        orient_pt = data['map_point']['orientation'].contiguous()
        pos_pl = data['map_polygon']['position'][:, :self.input_dim].contiguous()
        orient_pl = data['map_polygon']['orientation'].contiguous()
        orient_vector_pl = torch.stack([orient_pl.cos(), orient_pl.sin()], dim=-1)

        if self.dataset == 'argoverse_v2':
            if self.input_dim == 2:
                x_pt = data['map_point']['magnitude'].unsqueeze(-1)
                x_pl = None
            elif self.input_dim == 3:
                x_pt = torch.stack([data['map_point']['magnitude'], data['map_point']['height']], dim=-1)
                x_pl = None
            else:
                raise ValueError('{} is not a valid dimension'.format(self.input_dim))
            x_pt_categorical_embs = [self.type_pt_emb(data['map_point']['type'].long()),
                                     self.side_pt_emb(data['map_point']['side'].long())]
            x_pl_categorical_embs = [self.type_pl_emb(data['map_polygon']['type'].long()),
                                     self.int_pl_emb(data['map_polygon']['is_intersection'].long())]
        elif self.dataset == 'waymo':
            if self.input_dim == 2:
                x_pt = data['map_point']['magnitude'].unsqueeze(-1)
                x_pl = data['map_polygon']['speed_limit'].unsqueeze(-1)
            elif self.input_dim == 3:
                x_pt = torch.stack([data['map_point']['magnitude'], data['map_point']['height']], dim=-1)
                x_pl = data['map_polygon']['speed_limit'].unsqueeze(-1)
            else:
                raise ValueError('{} is not a valid dimension'.format(self.input_dim))
            x_pt_categorical_embs = [self.type_pt_emb(data['map_point']['type'].long()),
                                     self.side_pt_emb(data['map_point']['side'].long())]
            x_pl_categorical_embs = [self.type_pl_emb(data['map_polygon']['type'].long())]
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))
        x_pt = self.x_pt_emb(continuous_inputs=x_pt, categorical_embs=x_pt_categorical_embs)
        x_pl = self.x_pl_emb(continuous_inputs=x_pl, categorical_embs=x_pl_categorical_embs)

        edge_index_pt2pl = data['map_point', 'to', 'map_polygon']['edge_index']
        rel_pos_pt2pl = pos_pt[edge_index_pt2pl[0]] - pos_pl[edge_index_pt2pl[1]]
        rel_orient_pt2pl = wrap_angle(orient_pt[edge_index_pt2pl[0]] - orient_pl[edge_index_pt2pl[1]])
        if self.input_dim == 2:
            r_pt2pl = torch.stack(
                [torch.norm(rel_pos_pt2pl[:, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=orient_vector_pl[edge_index_pt2pl[1]],
                                          nbr_vector=rel_pos_pt2pl[:, :2]),
                 rel_orient_pt2pl], dim=-1)
        elif self.input_dim == 3:
            r_pt2pl = torch.stack(
                [torch.norm(rel_pos_pt2pl[:, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=orient_vector_pl[edge_index_pt2pl[1]],
                                          nbr_vector=rel_pos_pt2pl[:, :2]),
                 rel_pos_pt2pl[:, -1],
                 rel_orient_pt2pl], dim=-1)
        else:
            raise ValueError('{} is not a valid dimension'.format(self.input_dim))
        r_pt2pl = self.r_pt2pl_emb(continuous_inputs=r_pt2pl, categorical_embs=None)

        edge_index_pl2pl = data['map_polygon', 'to', 'map_polygon']['edge_index']
        edge_index_pl2pl_radius = radius_graph(x=pos_pl[:, :2], r=self.pl2pl_radius,
                                               batch=data['map_polygon']['batch'] if isinstance(data, Batch) else None,
                                               loop=False, max_num_neighbors=300)
        type_pl2pl = data['map_polygon', 'to', 'map_polygon']['type']
        type_pl2pl_radius = type_pl2pl.new_zeros(edge_index_pl2pl_radius.size(1), dtype=torch.uint8)
        edge_index_pl2pl, type_pl2pl = merge_edges(edge_indices=[edge_index_pl2pl_radius, edge_index_pl2pl],
                                                   edge_attrs=[type_pl2pl_radius, type_pl2pl], reduce='max')
        rel_pos_pl2pl = pos_pl[edge_index_pl2pl[0]] - pos_pl[edge_index_pl2pl[1]]
        rel_orient_pl2pl = wrap_angle(orient_pl[edge_index_pl2pl[0]] - orient_pl[edge_index_pl2pl[1]])
        if self.input_dim == 2:
            r_pl2pl = torch.stack(
                [torch.norm(rel_pos_pl2pl[:, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=orient_vector_pl[edge_index_pl2pl[1]],
                                          nbr_vector=rel_pos_pl2pl[:, :2]),
                 rel_orient_pl2pl], dim=-1)
        elif self.input_dim == 3:
            r_pl2pl = torch.stack(
                [torch.norm(rel_pos_pl2pl[:, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=orient_vector_pl[edge_index_pl2pl[1]],
                                          nbr_vector=rel_pos_pl2pl[:, :2]),
                 rel_pos_pl2pl[:, -1],
                 rel_orient_pl2pl], dim=-1)
        else:
            raise ValueError('{} is not a valid dimension'.format(self.input_dim))
        r_pl2pl = self.r_pl2pl_emb(continuous_inputs=r_pl2pl, categorical_embs=[self.type_pl2pl_emb(type_pl2pl.long())])

        for i in range(self.num_layers):
            x_pl = self.pt2pl_layers[i]((x_pt, x_pl), r_pt2pl, edge_index_pt2pl)
            x_pl = self.pl2pl_layers[i](x_pl, r_pl2pl, edge_index_pl2pl)
        if self.dataset == 'waymo':
            stop_sign_mask = data['map_polygon']['stop_sign_mask']
            stop_sign_vector = data['map_polygon']['stop_sign_position'][:, :self.input_dim] - pos_pl
            if self.input_dim == 2:
                ss_pl = torch.stack(
                    [torch.norm(stop_sign_vector[:, :2], p=2, dim=-1),
                     angle_between_2d_vectors(ctr_vector=orient_vector_pl,
                                              nbr_vector=stop_sign_vector[:, :2])], dim=-1)
            elif self.input_dim == 3:
                ss_pl = torch.stack(
                    [torch.norm(stop_sign_vector[:, :2], p=2, dim=-1),
                     angle_between_2d_vectors(ctr_vector=orient_vector_pl,
                                              nbr_vector=stop_sign_vector[:, :2]),
                     stop_sign_vector[:, -1]], dim=-1)
            else:
                raise ValueError('{} is not a valid dimension'.format(self.input_dim))
            ss_pl = self.stop_sign_pl_emb(continuous_inputs=ss_pl, categorical_embs=None)
            ss_pl = torch.where(stop_sign_mask.unsqueeze(-1), ss_pl, self.no_stop_sign_pl_emb.weight)
            x_pl = x_pl + ss_pl
        x_pl = x_pl.repeat_interleave(repeats=self.num_historical_steps,
                                      dim=0).reshape(-1, self.num_historical_steps, self.hidden_dim)
        if self.dataset == 'waymo':
            traffic_signal_mask = data['map_polygon']['trafic_signal_mask'][:, :self.num_historical_steps]
            traffic_signal_state = data['map_polygon']['traffic_signal_state'][:, :self.num_historical_steps]
            stop_point_vector = (data['map_polygon']['stop_point'][:, :self.num_historical_steps, :self.input_dim] -
                                 pos_pl.unsqueeze(1))
            if self.input_dim == 2:
                t_pl = torch.stack(
                    [torch.norm(stop_point_vector[:, :, :2], p=2, dim=-1),
                     angle_between_2d_vectors(ctr_vector=orient_vector_pl.unsqueeze(1),
                                              nbr_vector=stop_point_vector[:, :, :2])], dim=-1)
            elif self.input_dim == 3:
                t_pl = torch.stack(
                    [torch.norm(stop_point_vector[:, :, :2], p=2, dim=-1),
                     angle_between_2d_vectors(ctr_vector=orient_vector_pl.unsqueeze(1),
                                              nbr_vector=stop_point_vector[:, :, :2]),
                     stop_point_vector[:, :, -1]], dim=-1)
            else:
                raise ValueError('{} is not a valid dimension'.format(self.input_dim))
            t_pl_categorical_embs = [self.tfc_sig_state_pl_emb(traffic_signal_state.long()).view(-1, self.hidden_dim)]
            t_pl = self.tfc_sig_pl_emb(continuous_inputs=t_pl.view(-1, t_pl.size(-1)),
                                       categorical_embs=t_pl_categorical_embs)
            t_pl = t_pl.reshape(-1, self.num_historical_steps, self.hidden_dim)
            t_pl = torch.where(traffic_signal_mask.unsqueeze(-1),
                               t_pl,
                               self.tfc_sig_state_pl_emb.weight[-1].reshape(1, 1, -1))
            x_pl = x_pl + t_pl

        return {'x_pt': x_pt, 'x_pl': x_pl}
