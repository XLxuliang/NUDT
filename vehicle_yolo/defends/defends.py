from typing import Any, Dict
import torch
import torch.nn as nn
import torch.optim as optim

import os
import yaml
from easydict import EasyDict

from ultralytics import YOLO

from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.data import build_yolo_dataset, ClassificationDataset, build_dataloader
from ultralytics.utils import TQDM, emojis
from ultralytics.utils.torch_utils import select_device

# from ultralytics.models.classify. import ClassificationValidator

from nudt_ultralytics.callbacks.callbacks import callbacks_dict

from utils.sse import sse_clean_samples_gen_validated

from defends.ipeg_compression import JpegCompression
from defends.jpeg_scale import JpegScale
from defends.neural_cleanse import NeuralCleanse
from defends.pgd_purifier import PGDPurifier
from defends.fgsm_denoise import FGSMDenoise

class defends:
    def __init__(self, cfg, args):
        self.args = args
        self.cfg = EasyDict(load_yaml(args.cfg_yaml))
        self.device = self.cfg.device
        if self.cfg.task == "detect":
            self.model = DetectionModel(cfg=self.cfg.model, ch=3, nc=self.args.class_number, verbose=self.cfg.verbose)
            # print(self.model)
            ckpt, file = torch_safe_load(self.cfg.pretrained)
            self.model.load(weights=ckpt["model"])
            sse_model_loaded(model_name=self.args.model, weight_path=self.cfg.pretrained)
            # for param in self.model.parameters():
            #     print(param)
            
            data = check_det_dataset(self.cfg.data)
            self.dataset = build_yolo_dataset(self.cfg, data.get(self.cfg.split), self.cfg.batch, data, mode="val", stride=self.model.stride)
            self.dataloader = build_dataloader(self.dataset, self.cfg.batch, self.cfg.workers, shuffle=False, rank=-1, drop_last=self.cfg.compile, pin_memory=False)
            
        elif self.cfg.task == "classify":
            self.model = ClassificationModel(cfg=self.cfg.model, ch=3, nc=self.args.class_number, verbose=self.cfg.verbose)
            # print(self.model)
            
            ckpt, file = torch_safe_load(self.cfg.pretrained)
            self.model.load(weights=ckpt["model"])
            sse_model_loaded(model_name=self.args.model, weight_path=self.cfg.pretrained)
            # for param in self.model.parameters():
            #     print(param)
            
            data = check_cls_dataset(self.cfg.data, split=self.cfg.split)
            self.dataset = ClassificationDataset_nudt(root=data.get(self.cfg.split), args=self.cfg, augment=self.cfg.augment, prefix=self.cfg.split)
            self.dataloader = build_dataloader(self.dataset, self.cfg.batch, self.cfg.workers, rank=-1)
        
        
        if args.defend_method == 'comp':
            self.defend = JpegCompression(
                                clip_values=(0, 255),
                                quality=50,
                                channels_first=False,
                                apply_fit=True,
                                apply_predict=True,
                                verbose=False,
                            )
        elif args.defend_method == 'scale':
            self.defend = JpegScale(
                                scale=0.9,
                                interp="bilinear"
                            )
        elif args.defend_method == 'neural_cleanse':
            self.defend = NeuralCleanse(kernel_size=3)
        elif args.defend_method == 'pgd':
            self.defend = PGDPurifier(steps=10, alpha=1.0, epsilon=8.0)
        elif args.defend_method == 'fgsm':
            self.defend = FGSMDenoise(epsilon=8.0)
        else:
            raise ValueError('Invalid defend method!')
        
    
    def classify_save_adv_image(self, adv_image, ori_image_file):
        # print(ori_image_file)
        ori_image_name = ori_image_file.split('/')[-1]
        ori_cls_flod = ori_image_file.split('/')[-2]
        ori_dataset_name = glob.glob(os.path.join(f'{self.args.input_path}/data', '*/'))[0].split('/')[-2]
        adv_image_flod = f'{self.cfg.save_dir}/adv_{ori_dataset_name}/{self.cfg.split}/{ori_cls_flod}'
        os.makedirs(adv_image_flod, exist_ok=True)
        adv_image_file = f'{adv_image_flod}/{ori_image_name}'
        
        from torchvision import transforms
        to_pil = transforms.ToPILImage()
        pil_image = to_pil(adv_image)
        pil_image.save(adv_image_file)
        sse_adv_samples_gen_validated(adv_image_file)
        
    def detect_save_adv_image(self, adv_image, ori_image_file):
        # print(ori_image_file)
        ori_image_name = ori_image_file.split('/')[-1]
        ori_image_flod = ori_image_file.split('/')[-3]
        ori_dataset_name = glob.glob(os.path.join(f'{self.args.input_path}/data', '*/'))[0].split('/')[-2]
        adv_image_flod = f'{self.cfg.save_dir}/adv_{ori_dataset_name}/{ori_image_flod}/{self.cfg.split}'
        os.makedirs(adv_image_flod, exist_ok=True)
        adv_image_file = f'{adv_image_flod}/{ori_image_name}'
        
        from torchvision import transforms
        to_pil = transforms.ToPILImage()
        pil_image = to_pil(adv_image)
        pil_image.save(adv_image_file)
        sse_adv_samples_gen_validated(adv_image_file)
        
        ori_label_flod = 'labels'
        ori_label_file = ori_image_file.replace(ori_image_flod, ori_label_flod).split('.')[0] + '.txt'
        ori_label_name = ori_label_file.split('/')[-1]
        adv_label_flod = f'{self.cfg.save_dir}/adv_{ori_dataset_name}/{ori_label_flod}/{self.cfg.split}'
        os.makedirs(adv_label_flod, exist_ok=True)
        adv_label_file = f'{adv_label_flod}/{ori_label_name}'
        os.system(f"cp {ori_label_file} {adv_label_file}")
        
        

    def classify_preprocess(self, img):
        """Convert input images to model-compatible tensor format with appropriate normalization."""
        if not isinstance(img, torch.Tensor):
            img = torch.stack(
                [self.transforms(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))) for im in img], dim=0
            )
        img = (img if isinstance(img, torch.Tensor) else torch.from_numpy(img)).to(self.model.device)
        return img.half() if self.cfg.half else img.float()  # Convert uint8 to fp16/32
    
    def detect_preprocess(self, im: torch.Tensor) -> torch.Tensor:
        """
        Prepare input image before inference.

        Args:
            im (torch.Tensor | list[np.ndarray]): Images of shape (N, 3, H, W) for tensor, [(H, W, 3) x N] for list.

        Returns:
            (torch.Tensor): Preprocessed image tensor of shape (N, 3, H, W).
        """
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = np.stack(self.pre_transform(im))
            if im.shape[-1] == 3:
                im = im[..., ::-1]  # BGR to RGB
            im = im.transpose((0, 3, 1, 2))  # BHWC to BCHW, (n, 3, h, w)
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im)

        im = im.to(self.device)
        im = im.half() if self.cfg.half else im.float()  # uint8 to fp16/32
        if not_tensor:
            im /= 255  # 0 - 255 to 0.0 - 1.0
        return im
    
    def run_defend(self):
        for batch_i, batch in enumerate(self.dataloader):
            if args.defend_method == 'comp':
                if self.cfg.task == 'classify':
                    images = self.classify_preprocess(im=batch["img"])
                    
                elif self.cfg.task == 'detect':
                    images = self.detect_preprocess(im=batch["img"])
                
                if self.cfg.task == 'classify':
                    preds = self.model.predict(x=adv_images)
                    loss = loss_fn(preds, labels)
                else:
                    batch['img'] = adv_images
                    preds = self.model.forward(x=batch["img"])
                    loss, loss_items = loss_fn(preds, batch) # loss[0]: box, loss[1]: cls, loss[2]: df1
                    # loss = loss.sum()
                    loss = loss[1]
            
            
            # print(batch.keys()) # dict_keys(['batch_idx', 'bboxes', 'cls', 'im_file', 'img', 'ori_shape', 'ratio_pad', 'resized_shape'])
            if self.args.attack_method == 'pgd':
                adv_images = self.pgd(batch, eps=self.args.epsilon, alpha=self.args.step_size, steps=self.args.max_iterations, random_start=self.args.random_start, loss_function=self.args.loss_function)
            elif self.args.attack_method == 'fgsm':
                adv_images = self.fgsm(batch, eps=self.args.epsilon, loss_function=self.args.loss_function)
            elif self.args.attack_method == 'cw':
                adv_images = self.cw(batch, c=1, kappa=0, steps=self.args.max_iterations, lr=self.args.lr, optimization_method=self.args.optimization_method)
            elif self.args.attack_method == 'bim':
                adv_images = self.bim(batch, eps=self.args.epsilon, alpha=self.args.step_size, steps=self.args.max_iterations, loss_function=self.args.loss_function)
            elif self.args.attack_method == 'deepfool':
                adv_images, _ = self.deepfool(batch, steps=self.args.max_iterations, overshoot=0.02)
            else:
                raise ValueError('Invalid attach method!')
            
            if self.cfg.task == "detect":
                self.detect_save_adv_image(adv_image=adv_images[0], ori_image_file=batch["im_file"][0])
            elif self.cfg.task == "classify":
                self.classify_save_adv_image(adv_image=adv_images[0], ori_image_file=batch["im_file"][0])
            if batch_i == self.args.gen_adv_sample_num - 1:
                break


    def run_defend(self, args):
        for batch_i, batch in enumerate(bar):
            batch = self.preprocess(batch)
            if args.defend_method == 'comp':
                #clean_image, _ = self.defend(batch["img"].numpy())
                #clean_image = tensor.from_numpy(clean_image)
                clean_image, _ = self.defend(batch["img"].detach().cpu().numpy())
                clean_image = torch.from_numpy(clean_image)
                
                if self.cfg.task == 'classify':
                    images = self.classify_preprocess(im=batch["img"])
                    labels = batch["cls"]
                    loss_fn = self.gen_loss_fn(loss_function)
                elif self.cfg.task == 'detect':
                    images = self.detect_preprocess(im=batch["img"])
                    loss_fn = v8DetectionLoss_nudt(self.model, self.cfg)
            
            elif args.defend_method == 'scale':
                clean_image, _ = self.defend(batch["img"])
                #clean_image, _ = self.defend(batch["img"].detach().cpu())
            elif args.defend_method in ['neural_cleanse', 'pgd', 'fgsm']:
                clean_image, _ = self.defend(batch["img"].detach().cpu())
            else:
                raise ValueError('Invalid defend method!')
            
        