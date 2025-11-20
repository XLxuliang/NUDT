import torch
import torch.nn.functional as F


def total_variation(x: torch.Tensor) -> torch.Tensor:
    """
    Simple anisotropic TV loss on NCHW tensor scaled in [0,255].
    """
    dh = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    dw = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    return dh + dw


class PGDPurifier:
    """
    Gradient-based purification on the input image via projected gradient descent
    to minimize a denoising objective (TV + fidelity to original).
    This does not require a model and serves as a generic defence.
    """

    def __init__(self, steps: int = 10, alpha: float = 1.0, epsilon: float = 8.0, tv_weight: float = 1.0, l2_weight: float = 0.01):
        self.steps = int(steps)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.tv_weight = float(tv_weight)
        self.l2_weight = float(l2_weight)

    def __call__(self, x, y=None):
        # Expect torch.Tensor NCHW or NHWC
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x)

        need_permute_back = False
        if x.ndim != 4:
            raise ValueError("Expect 4D tensor for images batch")
        if x.shape[-1] in (1, 3):
            x = x.permute(0, 3, 1, 2)
            need_permute_back = True

        x = x.detach().float().cpu()
        # assume inputs are in [0,255] or [0,1]; bring to [0,255]
        x_max = 255.0 if x.max() > 1.5 else 1.0
        scale = 255.0 / x_max
        x = x * scale

        ori = x.clone()
        z = x.clone().requires_grad_(True)
        for _ in range(self.steps):
            tv_loss = total_variation(z)
            l2_loss = F.mse_loss(z, ori)
            loss = self.tv_weight * tv_loss + self.l2_weight * l2_loss
            grad = torch.autograd.grad(loss, z, retain_graph=False, create_graph=False)[0]
            with torch.no_grad():
                z = z - self.alpha * grad.sign()
                z = torch.clamp(z, 0.0, 255.0)
                # Project to epsilon-ball around original in L_inf
                delta = torch.clamp(z - ori, min=-self.epsilon, max=self.epsilon)
                z = (ori + delta).detach().requires_grad_(True)

        out = z.detach().round().to(torch.uint8)
        if need_permute_back:
            out = out.permute(0, 2, 3, 1)
        return out, y


