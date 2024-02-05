"""SO(3) diffusion methods."""
import numpy as np
import os
import logging
import torch


def igso3_expansion(omega, eps, L=1000):
    """Truncated sum of IGSO(3) distribution.

    This function approximates the power series in equation 5 of
    "DENOISING DIFFUSION PROBABILISTIC MODELS ON SO(3) FOR ROTATIONAL
    ALIGNMENT"
    Leach et al. 2022

    This expression diverges from the expression in Leach in that here, eps =
    sqrt(2) * eps_leach, if eps_leach were the scale parameter of the IGSO(3).

    With this reparameterization, IGSO(3) agrees with the Brownian motion on
    SO(3) with t=eps^2.

    Args:
        omega: rotation of Euler vector (i.e. the angle of rotation)
        eps: std of IGSO(3).
        L: Truncation level
        use_torch: set true to use torch tensors, otherwise use numpy arrays.
    """

    
    ls = torch.arange(L).to(omega.device)

    if len(omega.shape) == 2:
        # Used during predicted score calculation.
        ls = ls[None, None]  # [1, 1, L]
        omega = omega[..., None]  # [num_batch, num_res, 1]
        eps = eps[..., None]
    elif len(omega.shape) == 1:
        # Used during cache computation.
        ls = ls[None]  # [1, L]
        omega = omega[..., None]  # [num_batch, 1]
    else:
        raise ValueError("Omega must be 1D or 2D.")
    p = (2*ls + 1) * torch.exp(-ls*(ls+1)*eps**2/2) * torch.sin(omega*(ls+1/2)) / torch.sin(omega/2)
    return p.sum(dim=-1)


def density(expansion, omega, marginal=True):
    """IGSO(3) density.

    Args:
        expansion: truncated approximation of the power series in the IGSO(3)
        density.
        omega: length of an Euler vector (i.e. angle of rotation)
        marginal: set true to give marginal density over the angle of rotation,
            otherwise include normalization to give density on SO(3) or a
            rotation with angle omega.
    """
    if marginal:
        # if marginal, density over [0, pi], else over SO(3)
        return expansion * (1-torch.cos(omega))/torch.pi
    else:
        # the constant factor doesn't affect any actual calculations though
        return expansion / 8 / torch.pi**2



def interp(x, xp, fp):
    # Ensure all tensors are of the same data type
    fp = fp.to(xp.dtype)
    x = x.to(xp.dtype)
    xp = xp.to(x.device)
    fp = fp.to(x.device)
    x = torch.round(x, decimals=5)

    # Find the indices for x values in the sorted y array
    indices = torch.searchsorted(xp, x)

    # Clip indices to ensure they are within the valid range
    indices = torch.clamp(indices, 0, xp.shape[-1] - 2)

    # Calculate the fractional difference between x and the nearest y values
    x_diff = (x - xp[indices]) / torch.clamp(xp[indices + 1] - xp[indices], min=1e-5)
    
    # Perform linear interpolation
    result = fp[indices] + x_diff * (fp[indices + 1] - fp[indices])
    return result



def compose_rotvec(ac, bc):
    alpha = torch.norm(ac, dim=-1)+1e-9
    beta = torch.norm(bc, dim=-1)+1e-9
    a = ac/alpha.unsqueeze(-1)
    b = bc/beta.unsqueeze(-1)

    sina = torch.sin(alpha)
    cosa = torch.cos(alpha)
    sinb = torch.sin(beta)
    cosb = torch.cos(beta)

    gamma = torch.arccos(cosa*cosb - sina*sinb*torch.einsum("...ik,...ik->...i", a, b))
    c = ((sina*cosb).unsqueeze(-1)*a + (cosa*sinb).unsqueeze(-1)*b + (sina*sinb).unsqueeze(-1)*torch.cross(a,b,dim=-1))
    c = c/(torch.norm(c, dim=-1)+1e-9).unsqueeze(-1)
    
    return gamma.unsqueeze(-1)*c




def score(exp, omega, eps, L=1000):  # score of density over SO(3)
    """score uses the quotient rule to compute the scaling factor for the score
    of the IGSO(3) density.

    This function is used within the Diffuser class to when computing the score
    as an element of the tangent space of SO(3).

    This uses the quotient rule of calculus, and take the derivative of the
    log:
        d hi(x)/lo(x) = (lo(x) d hi(x)/dx - hi(x) d lo(x)/dx) / lo(x)^2
    and
        d log expansion(x) / dx = (d expansion(x)/ dx) / expansion(x)

    Args:
        exp: truncated expansion of the power series in the IGSO(3) density
        omega: length of an Euler vector (i.e. angle of rotation)
        eps: scale parameter for IGSO(3) -- as in expansion() this scaling
            differ from that in Leach by a factor of sqrt(2).
        L: truncation level
        use_torch: set true to use torch tensors, otherwise use numpy arrays.

    Returns:
        The d/d omega log IGSO3(omega; eps)/(1-cos(omega))

    """

    
    ls = torch.arange(L)[None].to(omega.device)
    if len(omega.shape) == 2:
        ls = ls[None]
    elif len(omega.shape) > 2:
        raise ValueError("Omega must be 1D or 2D.")
    omega = omega[..., None]
    eps = eps[..., None]
    hi = torch.sin(omega * (ls + 1 / 2))
    dhi = (ls + 1 / 2) * torch.cos(omega * (ls + 1 / 2))
    lo = torch.sin(omega / 2)
    dlo = 1 / 2 * torch.cos(omega / 2)
    dSigma = (2 * ls + 1) * torch.exp(-ls * (ls + 1) * eps**2/2) * (lo * dhi - hi * dlo) / lo ** 2
    dSigma = dSigma.sum(dim=-1)
    
    return dSigma / (exp + 1e-4)


class SO3Diffuser:

    def __init__(self, so3_conf):
        self.schedule = so3_conf.schedule

        self.min_sigma = torch.tensor(so3_conf.min_sigma)
        self.max_sigma = torch.tensor(so3_conf.max_sigma)

        self.num_sigma = so3_conf.num_sigma
        self.use_cached_score = so3_conf.use_cached_score
        self._log = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Discretize omegas for calculating CDFs. Skip omega=0.
        self.discrete_omega = torch.linspace(0, torch.pi, so3_conf.num_omega+1)[1:]

        # Precompute IGSO3 values.
        replace_period = lambda x: str(x).replace('.', '_')
        cache_dir = os.path.join(
            so3_conf.cache_dir,
            f'eps_{so3_conf.num_sigma}_omega_{so3_conf.num_omega}_min_sigma_{replace_period(so3_conf.min_sigma)}_max_sigma_{replace_period(so3_conf.max_sigma)}_schedule_{so3_conf.schedule}'
        )

        # If cache directory doesn't exist, create it
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir)
        pdf_cache = os.path.join(cache_dir, 'pdf_vals.npy')
        cdf_cache = os.path.join(cache_dir, 'cdf_vals.npy')
        score_norms_cache = os.path.join(cache_dir, 'score_norms.npy')

        if os.path.exists(pdf_cache) and os.path.exists(cdf_cache) and os.path.exists(score_norms_cache):
            self._log.info(f'Using cached IGSO3 in {cache_dir}')
            self._pdf = torch.load(pdf_cache)
            self._cdf = torch.load(cdf_cache)
            self._score_norms = torch.load(score_norms_cache)
        else:
            self._log.info(f'Computing IGSO3. Saving in {cache_dir}')
            # compute the expansion of the power series
            exp_vals = torch.cat([igso3_expansion(self.discrete_omega.to(self.device), sigma.to(self.device)).unsqueeze(0) for sigma in self.discrete_sigma]).cpu()

            self._pdf  = torch.cat([density(x.to(self.device), self.discrete_omega.to(self.device), marginal=True).unsqueeze(0) for x in exp_vals]).cpu()
            self._cdf = torch.cat([(pdf.to(self.device).cumsum(dim=-1) / so3_conf.num_omega * torch.pi).unsqueeze(0) for pdf in self._pdf]).cpu()

            self._score_norms = torch.cat([score(exp_vals[i].to(self.device), self.discrete_omega.to(self.device), x.to(self.device)).unsqueeze(0) for i, x in enumerate(self.discrete_sigma)]).cpu()

            torch.save(self._pdf, pdf_cache)
            torch.save(self._cdf, cdf_cache)
            torch.save(self._score_norms, score_norms_cache)

        self._score_scaling = torch.sqrt(torch.abs(torch.sum(
            self._score_norms**2 * self._pdf, axis=-1) / torch.sum(self._pdf, axis=-1)
        )) / torch.sqrt(torch.tensor(3).float())

    @property
    def discrete_sigma(self):
        return self.sigma(
            torch.linspace(0.0, 1.0, self.num_sigma)
        )

    def sigma_idx(self, sigma):
        """Calculates the index for discretized sigma during IGSO(3) initialization."""
        return torch.bucketize(sigma, self.discrete_sigma.to(sigma.device)) - 1

    def sigma(self, t):
        """Extract \sigma(t) corresponding to chosen sigma schedule."""
        if torch.any(t < 0) or torch.any(t > 1):
            raise ValueError(f'Invalid t={t}')
        if self.schedule == 'logarithmic':
            return torch.log(t * torch.exp(self.max_sigma) + (1 - t) * torch.exp(self.min_sigma))
        else:
            raise ValueError(f'Unrecognize schedule {self.schedule}')

    def diffusion_coef(self, t):
        """Compute diffusion coefficient (g_t)."""
        if self.schedule == 'logarithmic':
            g_t = torch.sqrt(2 * (torch.exp(self.max_sigma) - torch.exp(self.min_sigma)) * self.sigma(t) / torch.exp(self.sigma(t)))
        else:
            raise ValueError(f'Unrecognize schedule {self.schedule}')
        return g_t

    def t_to_idx(self, t: np.ndarray):
        """Helper function to go from time t to corresponding sigma_idx."""
        return self.sigma_idx(self.sigma(t))

    def sample_igso3(
            self,
            t: float,
            n_samples: float=1):
        """Uses the inverse cdf to sample an angle of rotation from IGSO(3).

        Args:
            t: continuous time in [0, 1].
            n_samples: number of samples to draw.

        Returns:
            [n_samples] angles of rotation.
        """
        
        x = torch.rand(n_samples, device=t.device)
        
        return interp(x, self._cdf[self.t_to_idx(t)], self.discrete_omega)

    def sample(
            self,
            t: float,
            n_samples: float=1):
        """Generates rotation vector(s) from IGSO(3).

        Args:
            t: continuous time in [0, 1].
            n_sample: number of samples to generate.

        Returns:
            [n_samples, 3] axis-angle rotation vectors sampled from IGSO(3).
        """
        x = torch.randn((n_samples, 3), device=t.device)
        x /= torch.norm(x, dim=-1).unsqueeze(-1)
        return x * self.sample_igso3(t, n_samples=n_samples).unsqueeze(-1)

    def sample_ref(self, n_samples: float=1):
        return self.sample(torch.tensor(1), n_samples=n_samples)

    

    def score(
            self,
            vec: torch.tensor,
            t: torch.tensor,
            eps: float=1e-6,
        ):
        """Computes the score of IGSO(3) density as a rotation vector.

        Same as score function but uses pytorch and performs a look-up.

        Args:
            vec: [..., 3] array of axis-angle rotation vectors.
            t: continuous time in [0, 1].

        Returns:
            [..., 3] score vector in the direction of the sampled vector with
            magnitude given by _score_norms.
        """
        omega = (torch.linalg.norm(vec, dim=-1) + eps).to(vec.device)
        if self.use_cached_score:
            score_norms_t = self._score_norms[self.t_to_idx(t)]
            omega_idx = torch.bucketize(omega, self.discrete_omega[:-1])
            omega_scores_t = torch.gather(score_norms_t, 1, omega_idx)
        else:
            sigma = self.discrete_sigma.to(vec.device)[self.t_to_idx(t)]
            omega_vals = igso3_expansion(omega, sigma.unsqueeze(-1))
            omega_scores_t = score(omega_vals, omega, sigma.unsqueeze(-1))
        return omega_scores_t.unsqueeze(-1) * vec / (omega.unsqueeze(-1) + eps)

    def score_scaling(self, t: np.ndarray):
        """Calculates scaling used for scores during trianing."""
        return self._score_scaling[self.t_to_idx(t)]

    def forward_marginal(self, rot_0: np.ndarray, t: float):
        """Samples from the forward diffusion process at time index t.

        Args:
            rot_0: [..., 3] initial rotations.
            t: continuous time in [0, 1].

        Returns:
            rot_t: [..., 3] noised rotation vectors.
            rot_score: [..., 3] score of rot_t as a rotation vector.
        """
        n_samples = torch.cumprod(torch.tensor(rot_0.shape[:-1]), dim=0)[-1]
        sampled_rots = self.sample(t, n_samples=n_samples)
        rot_score = self.score(sampled_rots, t).reshape(rot_0.shape)
        
        # Right multiply.
        rot_t = compose_rotvec(rot_0, sampled_rots)
        return rot_t, sampled_rots

    def reverse(
            self,
            rot_t: np.ndarray,
            score_t: np.ndarray,
            t: float,
            dt: float,
            noise_scale: float=1.0,
            ):
        """Simulates the reverse SDE for 1 step using the Geodesic random walk.

        Args:
            rot_t: [..., 3] current rotations at time t.
            score_t: [..., 3] rotation score at time t.
            t: continuous time in [0, 1].
            dt: continuous step size in [0, 1].
            add_noise: set False to set diffusion coefficent to 0.
            mask: True indicates which residues to diffuse.

        Returns:
            [..., 3] rotation vector at next step.
        """
        
        g_t = self.diffusion_coef(t)
        z = noise_scale * torch.randn(score_t.shape, device=rot_t.device)
        perturb = (g_t ** 2) * score_t * dt + g_t * torch.sqrt(dt) * z

        # Right multiply.
        rot_t_1 = compose_rotvec(rot_t, perturb)
        return rot_t_1