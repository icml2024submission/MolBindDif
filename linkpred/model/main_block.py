import torch
import torch.nn as nn

from openfold.utils.rigid_utils import Rigid

from linkpred.model.ipa import InvariantPointAttention
from linkpred.model.simple_blocks import StructureModuleTransition, AngleResnet
from linkpred.model.edge_update import EdgeTransitionNew




class MainBlock(nn.Module):

    def __init__(self, model_conf):
        super(MainBlock, self).__init__()
        self._model_conf = model_conf
        self._ac_conf = model_conf.ac

        self.scale_pos = lambda x: x * self._ac_conf.coordinate_scaling
        self.scale_rigids = lambda x: x.apply_trans_fn(self.scale_pos)

        self.unscale_pos = lambda x: x / self._ac_conf.coordinate_scaling
        self.unscale_rigids = lambda x: x.apply_trans_fn(self.unscale_pos)

        self.trunk = nn.ModuleDict()

        for b in range(self._ac_conf.num_blocks):
            self.trunk[f'ipa_rr_{b}'] = InvariantPointAttention(self._ac_conf)

            self.trunk[f'ipa_ln_r_{b}'] = nn.LayerNorm(self._ac_conf.c_s)
            
            self.trunk[f's_transition_r_{b}'] = StructureModuleTransition(c=self._ac_conf.c_s)

            if b < self._ac_conf.num_blocks-1:
                # No edge update on the last block.
                self.trunk[f'z_rr_update_{b}'] = EdgeTransitionNew(ac_conf=self._ac_conf)

        self.angle_resnet = AngleResnet(self._ac_conf.c_s, self._ac_conf.c_resnet, self._ac_conf.no_resnet_blocks, self._ac_conf.no_angles)

        
    def forward(self, s_r, z_rr, T_r):

        T_r = Rigid.from_tensor_7(T_r)
        s_r_initial = torch.clone(s_r)

        # Main trunk
        T_r = self.scale_rigids(T_r)
        for b in range(self._ac_conf.num_blocks):
            
            ipa_embed_rr  = self.trunk[f'ipa_rr_{b}'](s_r, s_r, z_rr, T_r, T_r)
           
            s_r = self.trunk[f'ipa_ln_r_{b}'](s_r + ipa_embed_rr)

            s_r = self.trunk[f's_transition_r_{b}'](s_r)

            if b < self._ac_conf.num_blocks-1:
                z_rr = self.trunk[f'z_rr_update_{b}'](s_r, s_r, z_rr)
        
        T_r = self.unscale_rigids(T_r)

        unnormalized_angles, angles = self.angle_resnet(s_r, s_r_initial)

        model_out = {
            'angles': angles,
            'unnormalized_angles': unnormalized_angles
        }
        return model_out
