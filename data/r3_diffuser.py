"""R^3 diffusion methods."""
import numpy as np
import torch


class R3Diffuser:
    """VP-SDE diffuser class for translations."""

    def __init__(self, r3_conf):
        """
        Args:
            min_b: starting value in variance schedule.
            max_b: ending value in variance schedule.
        """
        self._r3_conf = r3_conf
        self.min_b = r3_conf.min_b
        self.max_b = r3_conf.max_b

    def _scale(self, x):
        return x * self._r3_conf.coordinate_scaling

    def _unscale(self, x):
        return x / self._r3_conf.coordinate_scaling

    def b_t(self, t):
        if torch.any(t < 0) or torch.any(t > 1):
            raise ValueError(f'Invalid t={t}')
        return 2.0*(32.0*t**15 + 2.0*t**3 + t)

    def diffusion_coef(self, t):
        """Time-dependent diffusion coefficient."""
        return torch.sqrt(self.b_t(t))

    def drift_coef(self, x, t):
        """Time-dependent drift coefficient."""
        return -1/2 * self.b_t(t) * x

    def sample_ref(self, n_samples: float=1):
        return torch.randn(n_samples, 3)

    def marginal_b_t(self, t):
        return 4.0*t**16 + t**4 + t**2

    def calc_trans_0(self, score_t, x_t, t):
        beta_t = self.marginal_b_t(t)
        beta_t = beta_t[..., None, None]
        exp_fn = torch.exp
        cond_var = 1 - exp_fn(-beta_t)
        return (score_t * cond_var + x_t) / exp_fn(-1/2*beta_t)

    def forward(self, x_t_1, t, num_t):
        
        x_t_1 = self._scale(x_t_1)
        b_t = (self.marginal_b_t(t) / num_t).to(x_t_1.device)
        z_t_1 = torch.randn(x_t_1.shape, device = x_t_1.device)
        x_t = torch.sqrt(1 - b_t) * x_t_1 + torch.sqrt(b_t) * z_t_1
        return x_t

    def distribution(self, x_t, score_t, t, dt):
        x_t = self._scale(x_t)
        g_t = self.diffusion_coef(t)
        f_t = self.drift_coef(x_t, t)
        std = g_t * torch.sqrt(dt)
        mu = x_t - (f_t - g_t**2 * score_t) * dt
        
        return mu, std

    def forward_marginal(self, x_0: np.ndarray, t: float):
        """Samples marginal p(x(t) | x(0)).

        Args:
            x_0: [..., n, 3] initial positions in Angstroms.
            t: continuous time in [0, 1].

        Returns:
            x_t: [..., n, 3] positions at time t in Angstroms.
            score_t: [..., n, 3] score at time t in scaled Angstroms.
        """
        
        x_0 = self._scale(x_0)
        x_t = torch.exp(-1/2*self.marginal_b_t(t)) * x_0 + torch.sqrt(1 - torch.exp(-self.marginal_b_t(t)))*torch.randn(x_0.shape, device=x_0.device)
        score_t = self.score(x_t, x_0, t)
        x_t = self._unscale(x_t)
        return x_t,  x_t-self._unscale(x_0)

    def score_scaling(self, t: float):
        return 1 / torch.sqrt(self.conditional_var(t))

    def reverse(
            self,
            *,
            x_t: np.ndarray,
            score_t: np.ndarray,
            t: float,
            dt: float,
            noise_scale: float=1.0,
        ):
        """Simulates the reverse SDE for 1 step

        Args:
            x_t: [..., 3] current positions at time t in angstroms.
            score_t: [..., 3] rotation score at time t.
            t: continuous time in [0, 1].
            dt: continuous step size in [0, 1].
            mask: True indicates which residues to diffuse.

        Returns:
            [..., 3] positions at next step t-1.
        """
        
        x_t = self._scale(x_t)
        g_t = self.diffusion_coef(t)
        f_t = self.drift_coef(x_t, t)
        z = noise_scale * torch.randn(score_t.shape, device=x_t.device)
        perturb = (f_t - g_t**2 * score_t) * dt + g_t * torch.sqrt(dt) * z

        x_t_1 = x_t - perturb
        x_t_1 = self._unscale(x_t_1)

        return x_t_1

    def conditional_var(self, t):
        """Conditional variance of p(xt|x0).

        Var[x_t|x_0] = conditional_var(t)*I

        """
        return 1 - torch.exp(-self.marginal_b_t(t))
        

    def score(self, x_t, x_0, t, scale=False):
        exp_fn = torch.exp
        
        if scale:
            x_t = self._scale(x_t)
            x_0 = self._scale(x_0)
        return -(x_t - exp_fn(-1/2*self.marginal_b_t(t)) * x_0) / self.conditional_var(t)
