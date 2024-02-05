"""Score network module."""
import torch
import math
from torch import nn
from torch.nn import functional as F
from model import main_block
import functools as fn
from data import utils as du


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


def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    # Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    assert len(timesteps.shape) == 1
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

class Embedder(nn.Module):

    def __init__(self, model_conf):
        super(Embedder, self).__init__()
        self._model_conf = model_conf
        self._embed_conf = model_conf.embed

        # Time step embedding
        time_embed_size = self._embed_conf.time_embed_size

        s_r_in = time_embed_size
        z_rr_in = time_embed_size
        z_rp_in = time_embed_size
        z_pr_in = time_embed_size

        # Sequence index embedding
        index_embed_size = self._embed_conf.index_embed_size

        s_r_in += index_embed_size
        s_p_in = index_embed_size
        z_rr_in += index_embed_size
        z_pp_in = index_embed_size

        

        # Residue type embedding
        res_embed_size = self._embed_conf.restype_embed_size

        s_r_in += res_embed_size
        s_p_in += res_embed_size

        # Bind embedding
        bind_embed_size = self._embed_conf.bind_embed_size

        s_p_in += bind_embed_size
        z_rp_in += bind_embed_size
        z_pr_in += bind_embed_size

        # Embedders
        s_r_embed_size = self._model_conf.s_r_embed_size
        s_p_embed_size = self._model_conf.s_p_embed_size

        self.s_r_embedder = nn.Sequential(
            nn.Linear(s_r_in, s_r_embed_size),
            nn.ReLU(),
            nn.Linear(s_r_embed_size, s_r_embed_size),
            nn.ReLU(),
            nn.Linear(s_r_embed_size, s_r_embed_size),
            nn.LayerNorm(s_r_embed_size),
        )
        self.s_p_embedder = nn.Sequential(
            nn.Linear(s_p_in, s_p_embed_size),
            nn.ReLU(),
            nn.Linear(s_p_embed_size, s_p_embed_size),
            nn.ReLU(),
            nn.Linear(s_p_embed_size, s_p_embed_size),
            nn.LayerNorm(s_p_embed_size),
        )

        if self._embed_conf.embed_self_conditioning:
            z_rr_in += self._embed_conf.num_bins
            z_rp_in += self._embed_conf.num_bins
            z_pr_in += self._embed_conf.num_bins

        z_pp_in += self._embed_conf.num_bins

        
        z_rr_embed_size = self._model_conf.z_rr_embed_size
        z_rp_embed_size = self._model_conf.z_rp_embed_size
        z_pr_embed_size = self._model_conf.z_rp_embed_size
        z_pp_embed_size = self._model_conf.z_rp_embed_size

        self.z_rr_embedder = nn.Sequential(
            nn.Linear(z_rr_in, z_rr_embed_size),
            nn.ReLU(),
            nn.Linear(z_rr_embed_size, z_rr_embed_size),
            nn.ReLU(),
            nn.Linear(z_rr_embed_size, z_rr_embed_size),
            nn.LayerNorm(z_rr_embed_size),
        )
        self.z_rp_embedder = nn.Sequential(
            nn.Linear(z_rp_in, z_rp_embed_size),
            nn.ReLU(),
            nn.Linear(z_rp_embed_size, z_rp_embed_size),
            nn.ReLU(),
            nn.Linear(z_rp_embed_size, z_rp_embed_size),
            nn.LayerNorm(z_rp_embed_size),
        )
        self.z_pr_embedder = nn.Sequential(
            nn.Linear(z_pr_in, z_pr_embed_size),
            nn.ReLU(),
            nn.Linear(z_pr_embed_size, z_pr_embed_size),
            nn.ReLU(),
            nn.Linear(z_pr_embed_size, z_pr_embed_size),
            nn.LayerNorm(z_pr_embed_size),
        )
        self.z_pp_embedder = nn.Sequential(
            nn.Linear(z_pp_in, z_pp_embed_size),
            nn.ReLU(),
            nn.Linear(z_pp_embed_size, z_pp_embed_size),
            nn.ReLU(),
            nn.Linear(z_pp_embed_size, z_pp_embed_size),
            nn.LayerNorm(z_pp_embed_size),
        )

        self.timestep_embedder = fn.partial(
            get_timestep_embedding,
            embedding_dim=self._embed_conf.time_embed_size
        )

        self.index_embedder = fn.partial(
            get_index_embedding,
            embed_size=self._embed_conf.index_embed_size
        )

        self.residue_embedder = nn.Sequential(
            nn.Embedding(28, res_embed_size),
            nn.LayerNorm(res_embed_size)
        )
        
        self.bind_embedder = nn.Sequential(
            nn.Embedding(2, bind_embed_size),
            nn.LayerNorm(bind_embed_size)
        )

    def forward(
            self,
            *,
            t,
            ridx_r,
            ridx_p,
            rtype_r,
            rtype_p,
            bind_p,
            sc_r,
            sc_p
        ):
        """Embeds a set of inputs

        Args:
            seq_idx: [..., N] Positional sequence index for each residue.
            t: Sampled t in [0, 1].
            fixed_mask: mask of fixed (motif) residues.
            self_conditioning_ca: [..., N, 3] Ca positions of self-conditioning
                input.

        Returns:
            node_embed: [B, N, D_node]
            edge_embed: [B, N, N, D_edge]
        """
        num_batch, num_res_r = ridx_r.shape
        num_batch, num_res_p = ridx_p.shape

        s_r_feats = []
        s_p_feats = []

        t_r_embed = torch.tile(self.timestep_embedder(t)[:, None, :], (1, num_res_r, 1))

        s_r_feats = [t_r_embed]
        z_rr_feats = [torch.tile(t_r_embed, (1, num_res_r, 1))]
        z_rp_feats = [torch.tile(t_r_embed, (1, num_res_p, 1))]
        z_pr_feats = [torch.tile(t_r_embed, (1, num_res_p, 1))]

        # Residue type features.
        res_r_encode = self.residue_embedder(rtype_r)
        s_r_feats.append(res_r_encode)
        s_p_feats = [self.residue_embedder(rtype_p)]

        # Positional index features.
        s_r_feats.append(self.index_embedder(ridx_r))
        s_p_feats.append(self.index_embedder(ridx_p))

        rel_seq_offset_r = ridx_r.unsqueeze(-1) - ridx_r.unsqueeze(-2)
        rel_seq_offset_r = rel_seq_offset_r.reshape([num_batch, num_res_r**2])
        z_rr_feats.append(self.index_embedder(rel_seq_offset_r))

        rel_seq_offset_p = ridx_p.unsqueeze(-1) - ridx_p.unsqueeze(-2)
        rel_seq_offset_p = rel_seq_offset_p.reshape([num_batch, num_res_p**2])
        z_pp_feats = [self.index_embedder(rel_seq_offset_p)]

        # Bind features.
        bind_rp_embed = self.bind_embedder(bind_p)
        s_p_feats.append(bind_rp_embed)
        z_rp_feats.append(torch.tile(bind_rp_embed.unsqueeze(-3), (1, num_res_r, 1, 1)).reshape(num_batch, num_res_r*num_res_p, -1))
        z_pr_feats.append(torch.tile(bind_rp_embed.unsqueeze(-2), (1, 1, num_res_r, 1)).reshape(num_batch, num_res_r*num_res_p, -1))

        # Self-conditioning distogram.
        if self._embed_conf.embed_self_conditioning:
            sc_dgram_rr = du.calc_distogram(
                sc_r,
                sc_r,
                self._embed_conf.min_bin,
                self._embed_conf.max_bin,
                self._embed_conf.num_bins,
            )
            z_rr_feats.append(sc_dgram_rr.reshape([num_batch, num_res_r**2, -1]))

            sc_dgram_rp = du.calc_distogram(
                sc_r,
                sc_p,
                self._embed_conf.min_bin,
                self._embed_conf.max_bin,
                self._embed_conf.num_bins,
            )
            z_rp_feats.append(sc_dgram_rp.reshape([num_batch, num_res_r*num_res_p, -1]))

            sc_dgram_pr = du.calc_distogram(
                sc_p,
                sc_r,
                self._embed_conf.min_bin,
                self._embed_conf.max_bin,
                self._embed_conf.num_bins,
            )
            z_pr_feats.append(sc_dgram_pr.reshape([num_batch, num_res_r*num_res_p, -1]))
        
        sc_dgram_pp = du.calc_distogram(
                sc_p,
                sc_p,
                self._embed_conf.min_bin,
                self._embed_conf.max_bin,
                self._embed_conf.num_bins,
            )
        z_pp_feats.append(sc_dgram_pp.reshape([num_batch, num_res_p*num_res_p, -1]))

        s_r_embed = self.s_r_embedder(torch.cat(s_r_feats, dim=-1).float())
        s_p_embed = self.s_p_embedder(torch.cat(s_p_feats, dim=-1).float())

        z_rr_embed = self.z_rr_embedder(torch.cat(z_rr_feats, dim=-1).float())
        z_rp_embed = self.z_rp_embedder(torch.cat(z_rp_feats, dim=-1).float())
        z_pr_embed = self.z_pr_embedder(torch.cat(z_pr_feats, dim=-1).float())
        z_pp_embed = self.z_pp_embedder(torch.cat(z_pp_feats, dim=-1).float())

        z_rr_embed = z_rr_embed.reshape([num_batch, num_res_r, num_res_r, -1])
        z_rp_embed = z_rp_embed.reshape([num_batch, num_res_r, num_res_p, -1])
        z_pr_embed = z_pr_embed.reshape([num_batch, num_res_p, num_res_r, -1])
        z_pp_embed = z_pp_embed.reshape([num_batch, num_res_p, num_res_p, -1])

        return s_r_embed, s_p_embed, z_rr_embed, z_rp_embed, z_pr_embed, z_pp_embed


class ScoreNetwork(nn.Module):

    def __init__(self, model_conf, diffuser):
        super(ScoreNetwork, self).__init__()
        self._model_conf = model_conf

        self.embedding_layer = Embedder(model_conf)
        self.diffuser = diffuser
        self.score_model = main_block.MainBlock(model_conf, diffuser)

    def _apply_mask(self, aatype_diff, aatype_0, diff_mask):
        return diff_mask * aatype_diff + (1 - diff_mask) * aatype_0

    def forward(self, input_feats):
        
        s_r_embed, s_p_embed, z_rr_embed, z_rp_embed, z_pr_embed, z_pp_embed = self.embedding_layer(
            t=input_feats['t'],
            ridx_r=input_feats['ridx_r'],
            ridx_p=input_feats['ridx_p'],
            rtype_r=input_feats['rtype_r'],
            rtype_p=input_feats['rtype_p'],
            bind_p=input_feats['bind_p'],
            sc_r=input_feats['sc_r'],
            sc_p=input_feats['sc_p']
        )
        
        model_out = self.score_model(s_r_embed, s_p_embed, z_rr_embed, z_rp_embed, z_pr_embed, z_pp_embed, input_feats['T_r_t'], input_feats['T_p_0'], input_feats['t'])

        pred_out = {
            'trans_update': model_out['trans_update'],
            'rot_update': model_out['rot_update'],
            'trans_score': model_out['trans_score'],
            'rot_score': model_out['rot_score']
        }
        rigids_pred = model_out['T_r_0']
        pred_out['T_r_0'] = rigids_pred.to_tensor_7()

        return pred_out
