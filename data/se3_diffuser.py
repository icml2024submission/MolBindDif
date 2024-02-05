"""SE(3) diffusion methods."""
from data import so3_diffuser
from data import r3_diffuser
from data import utils as du
from openfold.utils import rigid_utils as ru
import torch
import logging

def _extract_trans_rots(rigid: ru.Rigid):
    rot = rigid.get_rots().get_rotvec()
    tran = rigid.get_trans()
    return tran, rot

def _quats_from_rotvec(rotvec):
    angle = torch.norm(rotvec, dim=-1)
    axis = rotvec/angle.unsqueeze(-1)
    qs = torch.cos(angle/2).unsqueeze(-1)
    qv = torch.sin(angle/2).unsqueeze(-1)*axis

    return torch.cat((qs,qv), dim=-1)

def _assemble_rigid(rotvec, trans):
    quats = _quats_from_rotvec(rotvec)
    rots = ru.Rotation(quats=quats)
    return ru.Rigid(rots=rots,trans=trans)

class SE3Diffuser:

    def __init__(self, se3_conf):
        self._log = logging.getLogger(__name__)
        self._se3_conf = se3_conf

        self._diffuse_rot = se3_conf.diffuse_rot
        self._so3_diffuser = so3_diffuser.SO3Diffuser(self._se3_conf.so3)

        self._diffuse_trans = se3_conf.diffuse_trans
        self._r3_diffuser = r3_diffuser.R3Diffuser(self._se3_conf.r3)

    def forward_marginal(
            self,
            rigids_0: ru.Rigid,
            t: float,
        ):
        """
        Args:
            rigids_0: [..., N] openfold Rigid objects
            t: continuous time in [0, 1].

        Returns:
            rigids_t: [..., N] noised rigid. [..., N, 7] if as_tensor_7 is true. 
            trans_score: [..., N, 3] translation score
            rot_score: [..., N, 3] rotation score
            trans_score_norm: [...] translation score norm
            rot_score_norm: [...] rotation score norm
        """
        
        trans_0, rot_0 = _extract_trans_rots(rigids_0)

        
        rot_t, rot_update = self._so3_diffuser.forward_marginal(rot_0, t)    #remove updates probably
            

        trans_t, trans_update = self._r3_diffuser.forward_marginal(trans_0, t)
        
        if torch.isnan(rot_t.sum()): print('rot')
        if torch.isnan(trans_t.sum()): print('trans')
        rigids_t = _assemble_rigid(rot_t, trans_t)
        

        return {
            'T_r_t': rigids_t,
            'trans_update': trans_update,
            'rot_update': rot_update
        }

    def calc_trans_0(self, trans_score, trans_t, t):
        return self._r3_diffuser.calc_trans_0(trans_score, trans_t, t)
    

    def calc_trans_score(self, trans_t, trans_0, t, scale=True):
        return self._r3_diffuser.score(trans_t, trans_0, t, scale=scale), trans_t-trans_0

    def calc_rot_score(self, rots_t, rots_0, t):
        rots_0_inv = rots_0.invert()
        quats_0_inv = rots_0_inv.get_quats()
        quats_t = rots_t.get_quats()
        quats_0t = ru.quat_multiply(quats_0_inv, quats_t)
        rotvec_0t = du.quat_to_rotvec(quats_0t)
        return self._so3_diffuser.score(rotvec_0t, t), rotvec_0t


    def _apply_mask(self, x_diff, x_fixed, diff_mask):
        return diff_mask * x_diff + (1 - diff_mask) * x_fixed

    def trans_parameters(self, trans_t, score_t, t, dt, mask):
        return self._r3_diffuser.distribution(
            trans_t, score_t, t, dt, mask)

    def score_scaling(self, t):
        rot_score_scaling = self._so3_diffuser.score_scaling(t)
        trans_score_scaling = self._r3_diffuser.score_scaling(t)
        return rot_score_scaling, trans_score_scaling
    

    def reverse(
            self,
            rigid_t,
            rot_score,
            trans_score,
            t,
            dt,
            center: bool=True,
            noise_scale: float=1.0,
        ):
        """Reverse sampling function from (t) to (t-1).

        Args:
            rigid_t: [..., N] protein rigid objects at time t.
            rot_score: [..., N, 3] rotation score.
            trans_score: [..., N, 3] translation score.
            t: continuous time in [0, 1].
            dt: continuous step size in [0, 1].
            mask: [..., N] which residues to update.
            center: true to set center of mass to zero after step

        Returns:
            rigid_t_1: [..., N] protein rigid objects at time t-1.
        """
        trans_t, rot_t = _extract_trans_rots(rigid_t)
        
        if not self._diffuse_rot:
            rot_t_1 = rot_t
        else:
            rot_t_1 = self._so3_diffuser.reverse(
                rot_t=rot_t,
                score_t=rot_score,
                t=t,
                dt=dt,
                noise_scale=noise_scale,
                )
        if not self._diffuse_trans:
            trans_t_1 = trans_t
        else:
            trans_t_1 = self._r3_diffuser.reverse(
                x_t=trans_t,
                score_t=trans_score,
                t=t,
                dt=dt,
                noise_scale=noise_scale
                )

        return _assemble_rigid(rot_t_1, trans_t_1)

    def sample_ref(
            self,
            n_samples
        ):
        """Samples rigids from reference distribution.

        Args:
            n_samples: Number of samples.
            impute: Rigid objects to use as imputation values if either
                translations or rotations are not diffused.
        """

        rot_ref = self._so3_diffuser.sample_ref(n_samples=n_samples)
        
        trans_ref = self._r3_diffuser.sample_ref(n_samples=n_samples)
        

        
        trans_ref = self._r3_diffuser._unscale(trans_ref)
        rigids_t = _assemble_rigid(rot_ref, trans_ref)
        
        return {'T_r_t': rigids_t}
