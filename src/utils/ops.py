from src import _C
from torch.autograd import Function


class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz, npoint, seeds=None):
        # type: (Any, torch.Tensor, int, torch.Tensor, bool) -> torch.Tensor
        r"""
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance with seeds

        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor where N > npoint
        npoint : int32
            number of features in the sampled set
        seeds : torch.Tensor
            (B, 3) tensor

        Returns
        -------
        torch.Tensor
            (B, npoint) tensor containing the set
        """
        if seeds is None:
            seeds = xyz.new_zeros(0)
            out = _C.furthest_point_sampling(xyz, npoint, seeds, False)
        else:
            out = _C.furthest_point_sampling(xyz, npoint, seeds, True)

        ctx.mark_non_differentiable(out)

        return out

    @staticmethod
    def backward(ctx, grad_out):
        return ()


furthest_point_sample = FurthestPointSampling.apply


class ComputeRBF(Function):
    @staticmethod
    def forward(ctx, points, centers, rotates, scales):
        r"""
        Parameters
        ----------
        points : torch.Tensor (B, N, 3) 
        centers : torch.Tensor (B, nk, 3) 
        rotates : torch.Tensor (B, nk, 9) 
        scales : torch.Tensor (B, nk, 3) 

        Returns
        -------
        torch.Tensor (B, N, nk) 
        """
        out = _C.compute_rbf(points, centers, rotates, scales)

        ctx.mark_non_differentiable(out)

        return out

    @staticmethod
    def backward(ctx, grad_out):
        return ()


compute_rbf = ComputeRBF.apply