import torch
import torch.nn.functional as F


def total_variation(x: torch.Tensor) -> torch.Tensor:
    dh = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    dw = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    return dh + dw


class FGSMDenoise:
    """
    Single-step gradient denoising (FGSM-style) on a TV + L2 fidelity objective.
    """

    def __init__(self, epsilon: float = 8.0, tv_weight: float = 1.0, l2_weight: float = 0.01):
        self.epsilon = float(epsilon)
        self.tv_weight = float(tv_weight)
        self.l2_weight = float(l2_weight)

    def __call__(self, x, y=None):
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x)

        need_permute_back = False
        if x.ndim != 4:
            raise ValueError("Expect 4D tensor for images batch")
        if x.shape[-1] in (1, 3):
            x = x.permute(0, 3, 1, 2)
            need_permute_back = True

        x = x.detach().float().cpu()
        x_max = 255.0 if x.max() > 1.5 else 1.0
        scale = 255.0 / x_max
        x = x * scale

        ori = x.clone()
        z = x.clone().requires_grad_(True)
        tv_loss = total_variation(z)
        l2_loss = F.mse_loss(z, ori)
        loss = self.tv_weight * tv_loss + self.l2_weight * l2_loss
        # print('==================')
        # print(loss)
        # print(z)
        # print('--------------')
        grad = torch.autograd.grad(loss, z, retain_graph=False, create_graph=False)[0]
        with torch.no_grad():
            z = z - self.epsilon * grad.sign()
            z = torch.clamp(z, 0.0, 255.0)
            delta = torch.clamp(z - ori, min=-self.epsilon, max=self.epsilon)
            z = (ori + delta)

        out = z.detach().round().to(torch.uint8)
        if need_permute_back:
            out = out.permute(0, 2, 3, 1)
        
        # print('+'*100)
        # print(out)
        return out, y


