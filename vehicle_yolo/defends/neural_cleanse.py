import numpy as np
import torch
from PIL import Image, ImageFilter


class NeuralCleanse:
    """
    Lightweight input purification inspired by Neural Cleanse ideas:
    apply a median filter to suppress small, localized triggers.

    This is a preprocessing defence (no model required).
    """

    def __init__(self, kernel_size: int = 3):
        assert kernel_size % 2 == 1 and kernel_size >= 3, "kernel_size must be odd and >=3"
        self.kernel_size = kernel_size

    def __call__(self, x, y=None):
        # Accept torch.Tensor (N, C, H, W) or (N, H, W, C) or numpy with same
        need_permute_back = False
        is_numpy = isinstance(x, np.ndarray)
        if is_numpy:
            arr = x
            if arr.ndim != 4:
                raise ValueError("Expect 4D input for images batch")
            if arr.shape[-1] in (1, 3):
                # NHWC
                batch = arr
            else:
                # NCHW -> NHWC
                batch = np.transpose(arr, (0, 2, 3, 1))
                need_permute_back = True
        else:
            t = x.detach().cpu()
            if t.ndim != 4:
                raise ValueError("Expect 4D tensor for images batch")
            if t.shape[-1] in (1, 3):
                batch = t.numpy()
            else:
                batch = t.permute(0, 2, 3, 1).numpy()
                need_permute_back = True

        batch = batch.astype(np.uint8) if batch.dtype != np.uint8 else batch

        cleaned_list = []
        for i in range(batch.shape[0]):
            img = Image.fromarray(batch[i])
            img = img.filter(ImageFilter.MedianFilter(size=self.kernel_size))
            cleaned_list.append(np.array(img, dtype=np.uint8))
        cleaned = np.stack(cleaned_list, axis=0)

        if need_permute_back:
            cleaned = np.transpose(cleaned, (0, 3, 1, 2))
        return (torch.from_numpy(cleaned) if not is_numpy else cleaned), y


