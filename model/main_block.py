"""Fork of Openfold's IPA."""
import torch
import torch.nn as nn

from openfold.utils.rigid_utils import Rigid

from model.primitives import Linear
from model.ipa import InvariantPointAttention
from model.simple_blocks import StructureModuleTransition, BackboneUpdate
from model.edge_update import EdgeTransition, EdgeTransitionNew, EdgeTransitionTri




class MainBlock(nn.Module):

    def __init__(self, model_conf, diffuser):
        super(MainBlock, self).__init__()
        self._model_conf = model_conf
        self._ac_conf = model_conf.ac

        self.diffuser = diffuser

        self.scale_pos = lambda x: x * self._ac_conf.coordinate_scaling
        self.scale_rigids = lambda x: x.apply_trans_fn(self.scale_pos)

        self.unscale_pos = lambda x: x / self._ac_conf.coordinate_scaling
        self.unscale_rigids = lambda x: x.apply_trans_fn(self.unscale_pos)

        self.trunk = nn.ModuleDict()

        for b in range(self._ac_conf.num_blocks):
            self.trunk[f'ipa_rr_{b}'] = InvariantPointAttention(self._ac_conf)
            self.trunk[f'ipa_rp_{b}'] = InvariantPointAttention(self._ac_conf)
            if b < self._ac_conf.num_blocks-1:
                self.trunk[f'ipa_pr_{b}'] = InvariantPointAttention(self._ac_conf)
                self.trunk[f'ipa_pp_{b}'] = InvariantPointAttention(self._ac_conf)

            self.trunk[f'ipa_ln_r_{b}'] = nn.LayerNorm(self._ac_conf.c_s)
            if b < self._ac_conf.num_blocks-1:
                self.trunk[f'ipa_ln_p_{b}'] = nn.LayerNorm(self._ac_conf.c_s)
            
            self.trunk[f's_transition_r_{b}'] = StructureModuleTransition(c=self._ac_conf.c_s)
            if b < self._ac_conf.num_blocks-1:
                self.trunk[f's_transition_p_{b}'] = StructureModuleTransition(c=self._ac_conf.c_s)
            
            self.trunk[f'bb_update_{b}'] = BackboneUpdate(self._ac_conf.c_s)

            if b < self._ac_conf.num_blocks-1:
                # No edge update on the last block.
                self.trunk[f'z_rr_update_{b}'] = EdgeTransitionNew(ac_conf=self._ac_conf)
                self.trunk[f'z_rp_update_{b}'] = EdgeTransitionNew(ac_conf=self._ac_conf)
                self.trunk[f'z_pr_update_{b}'] = EdgeTransitionNew(ac_conf=self._ac_conf)
                self.trunk[f'z_pp_update_{b}'] = EdgeTransitionNew(ac_conf=self._ac_conf)
                '''self.trunk[f'z_update_{b}'] = EdgeTransitionTri(ac_conf=self._ac_conf)'''


        
    def forward(self, s_r, s_p, z_rr, z_rp, z_pr, z_pp, T_r_f, T_p_f, t):

        init_T_r = Rigid.from_tensor_7(torch.clone(T_r_f))
        T_r = Rigid.from_tensor_7(torch.clone(T_r_f))
        T_p = Rigid.from_tensor_7(torch.clone(T_p_f))

        # Main trunk
        T_r = self.scale_rigids(T_r)
        T_p = self.scale_rigids(T_p)
        for b in range(self._ac_conf.num_blocks):
            
            ipa_embed_rr  = self.trunk[f'ipa_rr_{b}'](s_r, s_r, z_rr, T_r, T_r)
            ipa_embed_rp = self.trunk[f'ipa_rp_{b}'](s_r, s_p, z_rp, T_r, T_p)
            if b < self._ac_conf.num_blocks-1:
                ipa_embed_pr = self.trunk[f'ipa_pr_{b}'](s_p, s_r, z_pr, T_p, T_r)
                ipa_embed_pp = self.trunk[f'ipa_pp_{b}'](s_p, s_p, z_pp, T_p, T_p)
           
            s_r = self.trunk[f'ipa_ln_r_{b}'](s_r + ipa_embed_rr + ipa_embed_rp)
            if b < self._ac_conf.num_blocks-1:
                s_p = self.trunk[f'ipa_ln_p_{b}'](s_p + ipa_embed_pr + ipa_embed_pp)

            s_r = self.trunk[f's_transition_r_{b}'](s_r)
            if b < self._ac_conf.num_blocks-1:
                s_p = self.trunk[f's_transition_p_{b}'](s_p)
            
            T_r_update = self.trunk[f'bb_update_{b}'](s_r)
            T_r = T_r.compose_q_update_vec(T_r_update)

            if b < self._ac_conf.num_blocks-1:
                z_rr = self.trunk[f'z_rr_update_{b}'](s_r, s_r, z_rr)
                z_rp = self.trunk[f'z_rp_update_{b}'](s_r, s_p, z_rp)
                z_pr = self.trunk[f'z_pr_update_{b}'](s_p, s_r, z_pr)
                z_pp = self.trunk[f'z_pp_update_{b}'](s_p, s_p, z_pp)
                '''z_rr, z_rp, z_pr, z_pp = self.trunk[f'z_update_{b}'](s_r, s_p, z_rr, z_rp, z_pr, z_pp)'''
        
        T_r = self.unscale_rigids(T_r)
        T_p = self.unscale_rigids(T_p)
        rot_score, rot_update = self.diffuser.calc_rot_score(init_T_r.get_rots(), T_r.get_rots(), t)
        trans_score, trans_update = self.diffuser.calc_trans_score(init_T_r.get_trans(), T_r.get_trans(), t[:, None, None])


        model_out = {
            'T_r_0': T_r,
            'trans_score': trans_score,
            'rot_score': rot_score,
            'trans_update': trans_update,
            'rot_update': rot_update
        }
        return model_out
