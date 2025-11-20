import torch
import torch.nn.functional as F

# from art.defences.preprocessor.preprocessor import Preprocessor

class JpegScale():
    """
    图片放缩防御：先缩小再放大，滤掉高频对抗噪声。
    兼容 ART 的 Preprocessor 接口。
    """
    # params = ["scale", "interp"]

    def __init__(self, scale=0.5, interp="bilinear"):
        """
        :param scale:   缩小比例，0<scale<1
        :param interp:  插值方法，"bilinear" 或 "nearest"
        """
        # super().__init__()
        self.scale = scale
        self.interp = interp
        self._check_params()

    def __call__(self, x, y=None):
        """
        x: 输入张量，形状 (N, H, W, C) 或 (N, C, H, W) 均可
        返回: 同形状张量
        """

        # 统一转成 PyTorch 张量 (N, C, H, W)
        if x.ndim == 4 and x.shape[-1] in [1, 3]:   # (N, H, W, C)
            x = x.permute(0, 3, 1, 2)
            need_permute_back = True
        else:                                         # (N, C, H, W)
            x = x
            need_permute_back = False

        N, C, H, W = x.shape
        newH, newW = int(H * self.scale), int(W * self.scale)
        mode = dict(bilinear="bilinear", nearest="nearest")[self.interp]

        # 缩小
        x_small = F.interpolate(x, size=(newH, newW), mode=mode, align_corners=False)
        # 放大回原尺寸
        x_large = F.interpolate(x_small, size=(H, W), mode=mode, align_corners=False)

        # 转回uint8类型（关键修正）
        x_large = x_large.clamp(0, 255).to(torch.uint8)
        if need_permute_back:
            x_large = x_large.permute(0, 2, 3, 1)
        return x_large, y

    def _check_params(self):
        assert 0 < self.scale < 1, "scale 必须在 (0,1) 之间"
        assert self.interp in {"bilinear", "nearest"}