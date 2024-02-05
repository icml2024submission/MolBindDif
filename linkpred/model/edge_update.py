import torch
import torch.nn as nn
from linkpred.model.primitives import Linear, LayerNorm
from linkpred.model.axial_attention import RowSelfAttention, ColumnSelfAttention

from openfold.model.dropout import DropoutRowwise, DropoutColumnwise

from linkpred.model.triangular_attention import (
    TriangleAttentionStartingNode,
    TriangleAttentionEndingNode
)
from linkpred.model.triangular_multiplicative_update import (
    TriangleMultiplicationOutgoing,
    TriangleMultiplicationIncoming
)



class PairTransition(nn.Module):
    """
    Implements Algorithm 15.
    """

    def __init__(self, c_z, n):
        """
        Args:
            c_z:
                Pair transition channel dimension
            n:
                Factor by which c_z is multiplied to obtain hidden channel
                dimension
        """
        super(PairTransition, self).__init__()

        self.c_z = c_z
        self.n = n

        self.layer_norm = LayerNorm(self.c_z)
        self.linear_1 = Linear(self.c_z, self.n * self.c_z)
        self.relu = nn.ReLU()
        self.linear_2 = Linear(self.n * self.c_z, c_z)

    def _transition(self, z):
        # [*, N_res, N_res, C_hidden]
        z = self.linear_1(z)
        z = self.relu(z)

        # [*, N_res, N_res, C_z]
        z = self.linear_2(z)

        return z

    def forward(self, 
        z: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            z:
                [*, N_res, N_res, C_z] pair embedding
        Returns:
            [*, N_res, N_res, C_z] pair embedding update
        """
        

        z = self.layer_norm(z)

        z = self._transition(z=z)

        return z



class Communication(nn.Module):
    def __init__(self, c_s, c_z, c_hidden, eps=1e-3):
      
        super(Communication, self).__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.eps = eps

        self.layer_norm = nn.LayerNorm(self.c_hidden)
        self.linear_1 = Linear(self.c_s, self.c_hidden)
        self.linear_2 = Linear(self.c_s, self.c_hidden)
        self.linear_out = Linear(self.c_hidden * 2, self.c_z)

    def _cmct(self, a, b):
        _, num_res_a, _ = a.shape
        _, num_res_b, _ = b.shape

        res = torch.cat([
            torch.tile(a[:, :, None, :], (1, 1, num_res_b, 1)),
            torch.tile(b[:, None, :, :], (1, num_res_a, 1, 1)),
        ], axis=-1)

        return res


    def forward(self, 
        s1: torch.Tensor,
        s2: torch.Tensor
    ) -> torch.Tensor:

        s1 = self.layer_norm(self.linear_1(s1))
        s2 = self.layer_norm(self.linear_2(s2))

        cmct = self._cmct(s1, s2)

        norm = s1.shape[-2] * s2.shape[-2]
        cmct = cmct / (self.eps + norm)

        return self.linear_out(cmct)



class EdgeTransitionNew(nn.Module):
    def __init__(
            self,
            *,
            ac_conf
        ):
        super(EdgeTransitionNew, self).__init__()

        self.c_s = ac_conf.c_s
        self.c_z = ac_conf.c_z
        self.c_hidden_cmct = ac_conf.c_hidden_cmct
        self.c_hidden_pair_att = ac_conf.c_hidden_pair_att
        self.no_heads_pair = ac_conf.no_heads_pair
        self.transition_n = ac_conf.transition_n
     

        self.communication = Communication(self.c_s, self.c_z, self.c_hidden_cmct)

        self.column_attention = ColumnSelfAttention(self.c_z, self.no_heads_pair)

        self.row_attention = RowSelfAttention(self.c_z, self.no_heads_pair)

        self.pair_transition = PairTransition(self.c_z, self.transition_n)


    def forward(self, s1, s2, z):
        
        z = z + self.communication(s1, s2)

        z = z + self.column_attention(z)[0]

        z = z + self.row_attention(z)[0]

        z = z + self.pair_transition(z)

        return z
    


class EdgeTransitionTri(nn.Module):
    def __init__(
            self,
            *,
            ac_conf
        ):
        super(EdgeTransitionTri, self).__init__()

        self.c_s = ac_conf.c_s
        self.c_z = ac_conf.c_z
        self.c_hidden_cmct = ac_conf.c_hidden_cmct
        self.c_hidden_pair_att = ac_conf.c_hidden_pair_att
        self.no_heads_pair = ac_conf.no_heads_pair
        self.transition_n = ac_conf.transition_n
     

        self.communication = Communication(self.c_s, self.c_z, self.c_hidden_cmct)

        self.tri_mul_out = TriangleMultiplicationOutgoing(self.c_z, self.c_hidden_cmct)

        self.tri_mul_in = TriangleMultiplicationIncoming(self.c_z, self.c_hidden_cmct)

        self.tri_att_start = TriangleAttentionStartingNode(self.c_z, self.c_hidden_pair_att, self.no_heads_pair)

        self.tri_att_end = TriangleAttentionEndingNode(self.c_z, self.c_hidden_pair_att, self.no_heads_pair)

        self.ps_dropout_row_layer = DropoutRowwise(0.1)
        self.ps_dropout_col_layer = DropoutColumnwise(0.1)


        self.pair_transition = PairTransition(self.c_z, self.transition_n)


    def forward(self, s1, s2, z11, z12, z21, z22):
        
        s = torch.cat([s1, s2], dim=-2)

        num_res_1 = z11.shape[-2]

        z = torch.cat([torch.cat([z11, z12], dim=-2), torch.cat([z21, z22], dim=-2)], dim=-3)
        
        z = z + self.communication(s, s)

        z = z + self.ps_dropout_row_layer(self.tri_mul_out(z))

        z = z + self.ps_dropout_row_layer(self.tri_mul_in(z))

        z = z + self.ps_dropout_row_layer(self.tri_att_start(z))

        z = z + self.ps_dropout_col_layer(self.tri_att_end(z))
        
        z = z + self.pair_transition(z)

        return z[:,:num_res_1,:num_res_1], z[:,:num_res_1,num_res_1:], z[:,num_res_1:,:num_res_1], z[:,num_res_1:,num_res_1:]





class EdgeTransition(nn.Module):
    def __init__(
            self,
            *,
            node_embed_size,
            edge_embed_in,
            edge_embed_out,
            num_layers=2,
            node_dilation=2
        ):
        super(EdgeTransition, self).__init__()

        bias_embed_size = node_embed_size // node_dilation
        self.initial_embed_r = Linear(
            node_embed_size, bias_embed_size)
        self.initial_embed_p = Linear(
            node_embed_size, bias_embed_size)
        hidden_size = bias_embed_size * 2 + edge_embed_in
        trunk_layers = []
        for _ in range(num_layers):
            trunk_layers.append(Linear(hidden_size, hidden_size))
            trunk_layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*trunk_layers)
        self.final_layer = Linear(hidden_size, edge_embed_out)
        self.layer_norm = nn.LayerNorm(edge_embed_out)

    def forward(self, s1, s2, z):
        s1 = self.initial_embed_r(s1)
        s2 = self.initial_embed_p(s2)
        batch_size, num_res1, _ = s1.shape
        batch_size, num_res2, _ = s2.shape
        z_bias = torch.cat([
            torch.tile(s1[:, :, None, :], (1, 1, num_res2, 1)),
            torch.tile(s2[:, None, :, :], (1, num_res1, 1, 1)),
        ], axis=-1)

        z = torch.cat([z, z_bias], axis=-1).reshape(batch_size * num_res1*num_res2, -1)
        
        z = self.final_layer(self.trunk(z) + z)
        #z_rp = torch.utils.checkpoint.checkpoint(self.final_layer,self.trunk(z_rp) + z_rp)
        z = self.layer_norm(z)
        z = z.reshape(batch_size, num_res1, num_res2, -1)
        return z