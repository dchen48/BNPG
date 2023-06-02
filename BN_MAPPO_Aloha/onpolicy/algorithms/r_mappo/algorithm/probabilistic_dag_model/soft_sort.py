import torch
from torch import Tensor

### Sinkhorn soft sort ###

"""A PyTorch lib of ops with permutations, and sinkhorn balancing.
A PyTorch implementation of the library of operations and sampling with permutations
and their approximation with doubly-stochastic matrices, through Sinkhorn
balancing
Original reference implementation in tensorflow: https://github.com/google/gumbel_sinkhorn
Strongly inspired by https://github.com/HeddaCohenIndelman
"""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def sample_gumbel(shape, eps=1e-20):
    """Samples arbitrary-shaped standard gumbel variables.
    Args:
    shape: list of integers
    eps: float, for numerical stability
    Returns:
    A sample of standard Gumbel random variables
    """

    u = torch.rand(shape, device=device).float()
    return -torch.log(-torch.log(u + eps) + eps)


def matching(matrix_batch):
    """Solves a matching problem for a batch of matrices.
    This is a wrapper for the scipy.optimize.linear_sum_assignment function. It
    solves the optimization problem max_P sum_i,j M_i,j P_i,j with P a
    permutation matrix. Notice the negative sign; the reason, the original
    function solves a minimization problem
    Args:
    matrix_batch: A 3D tensor (a batch of matrices) with
        shape = [batch_size, N, N]. If 2D, the input is reshaped to 3D with
        batch_size = 1.
    Returns:
    listperms, a 2D integer tensor of permutations with shape [batch_size, N]
        so that listperms[n, :] is the permutation of range(N) that solves the
        problem  max_P sum_i,j M_i,j P_i,j with M = matrix_batch[n, :, :].
    """

    def hungarian(x):
        if x.ndim == 2:
            x = np.reshape(x, [1, x.shape[0], x.shape[1]])
        sol = np.zeros((x.shape[0], x.shape[1]), dtype=np.int32)
        for i in range(x.shape[0]):
            sol[i, :] = linear_sum_assignment(-x[i, :])[1].astype(np.int32)
        return sol

    listperms = hungarian(matrix_batch.detach().cpu().numpy())
    listperms = torch.from_numpy(listperms)
    return listperms


def sinkhorn(log_alpha, n_iters=20):
    """Performs incomplete Sinkhorn normalization to log_alpha.
    By a theorem by Sinkhorn and Knopp [1], a sufficiently well-behaved  matrix
    with positive entries can be turned into a doubly-stochastic matrix
    (i.e. its rows and columns add up to one) via the succesive row and column
    normalization.
    -To ensure positivity, the effective input to sinkhorn has to be
    exp(log_alpha) (elementwise).
    -However, for stability, sinkhorn works in the log-space. It is only at
    return time that entries are exponentiated.
    [1] Sinkhorn, Richard and Knopp, Paul.
    Concerning nonnegative matrices and doubly stochastic
    matrices. Pacific Journal of Mathematics, 1967
    Args:
    log_alpha: 2D tensor (a matrix of shape [N, N])
        or 3D tensor (a batch of matrices of shape = [batch_size, N, N])
    n_iters: number of sinkhorn iterations (in practice, as little as 20
        iterations are needed to achieve decent convergence for N~100)
    Returns:
    A 3D tensor of close-to-doubly-stochastic matrices (2D tensors are
        converted to 3D tensors with batch_size equals to 1)
    """
    n = log_alpha.size()[1]
    log_alpha = log_alpha.reshape(-1, n, n)

    for _ in range(n_iters):
        log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True)).reshape(-1, n, 1)
        log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True)).reshape(-1, 1, n)
    return torch.exp(log_alpha)


def gumbel_sinkhorn(log_alpha,
                    temp=1.0, noise_factor=1.0, n_iters=20,
                    squeeze=True, hard=False, fixed_noise = None):
    """Random doubly-stochastic matrices via gumbel noise.
    In the zero-temperature limit sinkhorn(log_alpha/temp) approaches
    a permutation matrix. Therefore, for low temperatures this method can be
    seen as an approximate sampling of permutation matrices, where the
    distribution is parameterized by the matrix log_alpha
    The deterministic case (noise_factor=0) is also interesting: it can be
    shown that lim t->0 sinkhorn(log_alpha/t) = M, where M is a
    permutation matrix, the solution of the
    matching problem M=arg max_M sum_i,j log_alpha_i,j M_i,j.
    Therefore, the deterministic limit case of gumbel_sinkhorn can be seen
    as approximate solving of a matching problem, otherwise solved via the
    Hungarian algorithm.
    Warning: the convergence holds true in the limit case n_iters = infty.
    Unfortunately, in practice n_iter is finite which can lead to numerical
    instabilities, mostly if temp is very low. Those manifest as
    pseudo-convergence or some row-columns to fractional entries (e.g.
    a row having two entries with 0.5, instead of a single 1.0)
    To minimize those effects, try increasing n_iter for decreased temp.
    On the other hand, too-low temperature usually lead to high-variance in
    gradients, so better not choose too low temperatures.
    Args:
    log_alpha: 2D tensor (a matrix of shape [N, N])
        or 3D tensor (a batch of matrices of shape = [batch_size, N, N])
    temp: temperature parameter, a float.
    noise_factor: scaling factor for the gumbel samples. Mostly to explore
        different degrees of randomness (and the absence of randomness, with
        noise_factor=0)
    n_iters: number of sinkhorn iterations. Should be chosen carefully, in
        inverse corresponde with temp to avoid numerical stabilities.
    squeeze: a boolean, if True and there is a single sample, the output will
        remain being a 3D tensor.
    hard: boolean
    Returns:
    sink: a 4D tensor of [batch_size, n_samples, N, N] i.e.
        batch_size *n_samples doubly-stochastic matrices. If n_samples = 1 and
        squeeze = True then the output is 3D.
    log_alpha_w_noise: a 4D tensor of [batch_size, n_samples, N, N] of
        noisy samples of log_alpha, divided by the temperature parameter. If
        n_samples = 1 then the output is 3D.
    """
    n = log_alpha.size()[1]
    batch_size = log_alpha.size()[0]

    if noise_factor == 0:
        noise = 0.0
    else:
        if fixed_noise == None:
            noise = sample_gumbel([batch_size, n, n]) * noise_factor
        else:
            noise = fixed_noise * noise_factor

    log_alpha_w_noise = log_alpha + noise
    log_alpha_w_noise = log_alpha_w_noise / temp

    log_alpha_w_noise_copy = log_alpha_w_noise.clone()
    sink = sinkhorn(log_alpha_w_noise_copy, n_iters)
    #ret = (sink, log_alpha_w_noise)
    ret = (sink, noise)

    if hard:
        # Straight through.
        hard_perms_inf = matching(log_alpha_w_noise)
        sink_hard = listperm2matperm(hard_perms_inf).to(device).float()
        #ret = (sink_hard - sink.detach() + sink, log_alpha_w_noise)
        ret = (sink_hard - sink.detach() + sink, noise)
    return ret


def listperm2matperm(listperm):
    """Converts a batch of permutations to its matricial form.
    Args:
    listperm: 2D tensor of permutations of shape [batch_size, n_objects] so that
      listperm[n] is a permutation of range(n_objects).
    Returns:
    a 3D tensor of permutations matperm of
      shape = [batch_size, n_objects, n_objects] so that matperm[n, :, :] is a
      permutation of the identity matrix, with matperm[n, i, listperm[n,i]] = 1
    """
    n_objects = listperm.size()[1]
    eye = np.eye(n_objects)[listperm]
    eye = torch.tensor(eye, dtype=torch.int32)
    return eye
