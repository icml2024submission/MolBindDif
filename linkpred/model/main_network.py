import torch
import math
from torch import nn
from linkpred.model import main_block
import functools as fn
from linkpred.data import utils as du




### SHOULD PAY ATTENTION TO THOSE EMBEDDERS

def get_index_embedding(indices, embed_size, max_len=2056):
    """Creates sine / cosine positional embeddings from a prespecified indices.

    Args:
        indices: offsets of size [..., N_edges] of type integer
        max_len: maximum length.
        embed_size: dimension of the embeddings to create

    Returns:
        positional embedding of shape [N, embed_size]
    """
    K = torch.arange(embed_size//2, device=indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size))).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size))).to(indices.device)
    pos_embedding = torch.cat([
        pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding


class Embedder(nn.Module):

    def __init__(self, model_conf):
        super(Embedder, self).__init__()
        self._model_conf = model_conf
        self._embed_conf = model_conf.embed

        # Sequence index embedding
        index_embed_size = self._embed_conf.index_embed_size

        s_r_in = index_embed_size
        z_rr_in = index_embed_size

        # Residue type embedding
        res_embed_size = self._embed_conf.restype_embed_size

        s_r_in += res_embed_size

        # Embedders
        s_r_embed_size = self._model_conf.s_r_embed_size

        self.s_r_embedder = nn.Sequential(
            nn.Linear(s_r_in, s_r_embed_size),
            nn.ReLU(),
            nn.Linear(s_r_embed_size, s_r_embed_size),
            nn.ReLU(),
            nn.Linear(s_r_embed_size, s_r_embed_size),
            nn.LayerNorm(s_r_embed_size),
        )

        z_rr_in += self._embed_conf.num_bins

        z_rr_embed_size = self._model_conf.z_rr_embed_size

        self.z_rr_embedder = nn.Sequential(
            nn.Linear(z_rr_in, z_rr_embed_size),
            nn.ReLU(),
            nn.Linear(z_rr_embed_size, z_rr_embed_size),
            nn.ReLU(),
            nn.Linear(z_rr_embed_size, z_rr_embed_size),
            nn.LayerNorm(z_rr_embed_size),
        )


        self.index_embedder = fn.partial(
            get_index_embedding,
            embed_size=self._embed_conf.index_embed_size
        )

        self.residue_embedder = nn.Sequential(
            nn.Embedding(8, res_embed_size),
            nn.LayerNorm(res_embed_size)
        )


    def forward(
            self,
            *,
            ridx_r,
            rtype_r,
            sc_r
        ):
   
        num_batch, num_res_r = ridx_r.shape
        
        s_r_feats = []

        # Residue type features.
        res_r_encode = self.residue_embedder(rtype_r)
        s_r_feats.append(res_r_encode)

        # Positional index features.
        s_r_feats.append(self.index_embedder(ridx_r))

        rel_seq_offset_r = ridx_r.unsqueeze(-1) - ridx_r.unsqueeze(-2)
        rel_seq_offset_r = rel_seq_offset_r.reshape([num_batch, num_res_r**2])
        z_rr_feats = [self.index_embedder(rel_seq_offset_r)]

        # Self-conditioning distogram.
        sc_dgram_rr = du.calc_distogram(
            sc_r,
            sc_r,
            self._embed_conf.min_bin,
            self._embed_conf.max_bin,
            self._embed_conf.num_bins,
        )
        z_rr_feats.append(sc_dgram_rr.reshape([num_batch, num_res_r**2, -1]))


        s_r_embed = self.s_r_embedder(torch.cat(s_r_feats, dim=-1).float())

        z_rr_embed = self.z_rr_embedder(torch.cat(z_rr_feats, dim=-1).float())

        z_rr_embed = z_rr_embed.reshape([num_batch, num_res_r, num_res_r, -1])

        return s_r_embed, z_rr_embed


class RestoreNetwork(nn.Module):

    def __init__(self, model_conf):
        super(RestoreNetwork, self).__init__()

        self._model_conf = model_conf
        self.embedding_layer = Embedder(model_conf)
        self.main_block = main_block.MainBlock(model_conf)


    def forward(self, input_feats):
        
        s_r_embed, z_rr_embed = self.embedding_layer(
            ridx_r=input_feats['ridx_r'],
            rtype_r=input_feats['rtype_r'],
            sc_r=input_feats['sc_r']
        )
        
        model_out = self.main_block(s_r_embed, z_rr_embed, input_feats['T_r'])

        pred_out = {
            'angles': model_out['angles'],
            'unnormalized_angles': model_out['unnormalized_angles']
        }

        return pred_out
