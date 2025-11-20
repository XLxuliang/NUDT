from typing import Any, Dict
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import os
import glob
import yaml
from easydict import EasyDict


from ultralytics import YOLO


from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.data import build_yolo_dataset, ClassificationDataset_nudt, build_dataloader
from ultralytics.utils import TQDM, emojis
from ultralytics.utils.torch_utils import select_device

# from ultralytics.models.yolo.classify.predict import ClassificationPredictor
from ultralytics.nn.tasks import ClassificationModel, DetectionModel
from ultralytics.nn.tasks import torch_safe_load
from ultralytics.utils.loss import v8DetectionLoss_nudt

from nudt_ultralytics.callbacks.callbacks import callbacks_dict
from ultralytics.utils import callbacks

from utils.sse import sse_adv_samples_gen_validated, sse_model_loaded
from utils.yaml_rw import load_yaml

class attacks:
    def __init__(self, args):
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
    
    def run_adv(self):
        for batch_i, batch in enumerate(self.dataloader):
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
            
    def gen_loss_fn(self, name):
        loss_fn = name.lower()
        if loss_fn == 'cross_entropy':
            loss_function = F.cross_entropy
        elif loss_fn == 'mse':
            loss_function = F.mse_loss
        elif loss_fn == 'l1':
            loss_function = F.l1_loss
        elif loss_fn == 'binary_cross_entropy':
            loss_function = F.binary_cross_entropy
        else:
            raise ValueError("Invalid Loss Type!")
        return loss_function
    
    def get_optimizer(self, name, param, lr):
        optimization_method = name.lower()
        if optimization_method == 'sgd':
            optimizer = torch.optim.SGD(param, lr=lr)
        elif self.hparams.optimizer.lower() == 'adam':
            optimizer = torch.optim.Adam(param, lr=lr)
        else:
            raise ValueError('Invalid optimizer type!')
###################################################################################################################################################
    

    def pgd(self, batch, eps=8 / 255, alpha=2 / 255, steps=10, random_start=True, loss_function='cross_entropy'):
        '''
        PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
        [https://arxiv.org/abs/1706.06083]

        Distance Measure : Linf
    
        Arguments:
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: True)

        Shape:
            - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
            - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
            - output: :math:`(N, C, H, W)`.
        '''

        if self.cfg.task == 'classify':
            images = self.classify_preprocess(im=batch["img"])
            labels = batch["cls"]
            loss_fn = self.gen_loss_fn(loss_function)
        elif self.cfg.task == 'detect':
            images = self.detect_preprocess(im=batch["img"])
            loss_fn = v8DetectionLoss_nudt(self.model, self.cfg)
        adv_images = images.clone().detach()
        
        if random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        
        for _ in range(steps):
            adv_images.requires_grad = True
            if self.cfg.task == 'classify':
                preds = self.model.predict(x=adv_images)
                loss = loss_fn(preds, labels)
            else:
                batch['img'] = adv_images
                preds = self.model.forward(x=batch["img"])
                loss, loss_items = loss_fn(preds, batch) # loss[0]: box, loss[1]: cls, loss[2]: df1
                # loss = loss.sum()
                loss = loss[1]
            
            # Update adversarial images
            grad = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]
            
            adv_images = adv_images.detach() + alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-eps, max=eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
    
    
    def fgsm(self, batch, eps=8 / 255, loss_function='cross_entropy'):
        '''
        FGSM in the paper 'Explaining and harnessing adversarial examples'
        [https://arxiv.org/abs/1412.6572]

        Distance Measure : Linf
    
        Arguments:
        eps (float): maximum perturbation. (Default: 8/255)

        Shape:
            - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
            - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
            - output: :math:`(N, C, H, W)`.
        '''
        if self.cfg.task == 'classify':
            batch['img'].requires_grad = True
            preds = self.model.predict(x=batch['img'])
            labels = batch["cls"]
            
            loss_fn = self.gen_loss_fn(loss_function)
            loss = loss_fn(preds, labels)
        else:
            batch['img'].requires_grad = True
            preds = self.model.forward(x=batch["img"])
            loss_fn = v8DetectionLoss_nudt(self.model, self.cfg)
            loss, loss_items = loss_fn(preds, batch) # loss[0]: box, loss[1]: cls, loss[2]: df1
            # loss = loss.sum()
            loss = loss[1]
            
        # Update adversarial images
        grad = torch.autograd.grad(loss, images, retain_graph=False, create_graph=False)[0]
        adv_images = images + eps * grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        return adv_images
    
    def cw(self, batch, c=1, kappa=0, steps=50, lr=0.01, loss_function='mse', optimization_method='adam'):
        '''
        CW in the paper 'Towards Evaluating the Robustness of Neural Networks'
        [https://arxiv.org/abs/1608.04644]

        Distance Measure : L2
    
        Arguments:
        c (float): c in the paper. parameter for box-constraint. (Default: 1)    
            :math:`minimize \Vert\frac{1}{2}(tanh(w)+1)-x\Vert^2_2+c\cdot f(\frac{1}{2}(tanh(w)+1))`
        kappa (float): kappa (also written as 'confidence') in the paper. (Default: 0)
            :math:`f(x')=max(max\{Z(x')_i:i\neq t\} -Z(x')_t, - \kappa)`
        steps (int): number of steps. (Default: 50)
        lr (float): learning rate of the Adam optimizer. (Default: 0.01)

        .. warning:: With default c, you can't easily get adversarial images. Set higher c like 1.

        Shape:
            - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
            - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
            - output: :math:`(N, C, H, W)`.
        '''
        if self.cfg.task == 'classify':
            images = self.classify_preprocess(im=batch["img"])
            labels = batch["cls"]
        elif self.cfg.task == 'detect':
            images = self.detect_preprocess(im=batch["img"])
            loss_fn = v8DetectionLoss_nudt(self.model, self.cfg)
        
        # w = torch.zeros_like(images).detach() # Requires 2x times
        w = self.inverse_tanh_space(images).detach()
        w.requires_grad = True

        best_adv_images = images.clone().detach()
        best_L2 = 1e10 * torch.ones((len(images))).to(self.device)
        prev_cost = 1e10
        dim = len(images.shape)

        # MSELoss = nn.MSELoss(reduction="none")
        MSELoss = self.gen_loss_fn(loss_function)
        Flatten = nn.Flatten()

        # optimizer = optim.Adam([w], lr=lr)
        optimizer = get_optimizer(optimization_method, [w], lr)

        for step in range(steps):
            # Get adversarial images
            adv_images = self.tanh_space(w)

            # Calculate loss
            current_L2 = MSELoss(Flatten(adv_images), Flatten(images)).sum(dim=1)
            L2_loss = current_L2.sum()

            if self.cfg.task == 'classify':
                preds = self.model.predict(x=adv_images)
                f_loss = self.f(preds, labels, kappa).sum()
            else:
                batch['img'] = adv_images
                preds = self.model.forward(x=batch["img"])
                loss, loss_items = loss_fn(preds, batch) # loss[0]: box, loss[1]: cls, loss[2]: df1
                # f_loss = loss.sum()
                f_loss = loss[1]


            cost = L2_loss + c * f_loss

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # Update adversarial images
            pre = torch.argmax(outputs.detach(), 1)
            # If the attack is not targeted we simply make these two values unequal
            condition = (pre != labels).float()

            # Filter out images that get either correct predictions or non-decreasing loss,
            # i.e., only images that are both misclassified and loss-decreasing are left
            mask = condition * (best_L2 > current_L2.detach())
            best_L2 = mask * current_L2.detach() + (1 - mask) * best_L2

            mask = mask.view([-1] + [1] * (dim - 1))
            best_adv_images = mask * adv_images.detach() + (1 - mask) * best_adv_images

            # Early stop when loss does not converge.
            # max(.,1) To prevent MODULO BY ZERO error in the next step.
            if step % max(steps // 10, 1) == 0:
                if cost.item() > prev_cost:
                    return best_adv_images
                prev_cost = cost.item()

        return best_adv_images

    def tanh_space(self, x):
        return 1 / 2 * (torch.tanh(x) + 1)

    def atanh(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))
    
    def inverse_tanh_space(self, x):
        # torch.atanh is only for torch >= 1.7.0
        # atanh is defined in the range -1 to 1
        return self.atanh(torch.clamp(x * 2 - 1, min=-1, max=1))

    # f-function in the paper
    def f(self, outputs, labels, kappa):
        one_hot_labels = torch.eye(outputs.shape[1]).to(self.device)[labels]

        # find the max logit other than the target class
        other = torch.max((1 - one_hot_labels) * outputs.to(self.device), dim=1)[0]
        # get the target class's logit
        real = torch.max(one_hot_labels * outputs.to(self.device), dim=1)[0]

        return torch.clamp((real - other), min=-kappa)
        
        
        
    def bim(self, batch, eps=8 / 255, alpha=2 / 255, steps=10, loss_function='cross_entropy'):
        '''
        BIM or iterative-FGSM in the paper 'Adversarial Examples in the Physical World'
        [https://arxiv.org/abs/1607.02533]

        Distance Measure : Linf
        
        Arguments:
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)

        .. note:: If steps set to 0, steps will be automatically decided following the paper.

        Shape:
            - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
            - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
            - output: :math:`(N, C, H, W)`.
        '''
        if steps == 0:
            steps = int(min(eps * 255 + 4, 1.25 * eps * 255))
        
        if self.cfg.task == 'classify':
            images = self.classify_preprocess(im=batch["img"])
            labels = batch["cls"]
            loss_fn = self.gen_loss_fn(loss_function)
        elif self.cfg.task == 'detect':
            images = self.detect_preprocess(im=batch["img"])
            loss_fn = v8DetectionLoss_nudt(self.model, self.cfg)
            
        ori_images = images.clone().detach()

        for _ in range(steps):
            images.requires_grad = True
            
            if self.cfg.task == 'classify':
                preds = self.model.predict(x=images)
                loss = loss_fn(preds, labels)
            else:
                batch['img'] = images
                preds = self.model.forward(x=batch["img"])
                loss, loss_items = loss_fn(preds, batch) # loss[0]: box, loss[1]: cls, loss[2]: df1
                # loss = loss.sum()
                loss = loss[1]
                
            # Update adversarial images
            grad = torch.autograd.grad(loss, images, retain_graph=False, create_graph=False)[0]

            adv_images = images + alpha * grad.sign()
            a = torch.clamp(ori_images - eps, min=0)
            b = (adv_images >= a).float() * adv_images + (adv_images < a).float() * a  # nopep8
            c = (b > ori_images + eps).float() * (ori_images + eps) + (b <= ori_images + eps).float() * b  # nopep8
            adv_images = torch.clamp(c, max=1).detach()

        return adv_images
    
    
    def deepfool(self, images, labels, steps=50, overshoot=0.02):
        '''
        'DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks'
        [https://arxiv.org/abs/1511.04599]
        Distance Measure : L2
        Arguments:
            steps (int): number of steps. (Default: 50)
            overshoot (float): parameter for enhancing the noise. (Default: 0.02)
        Shape:
            - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
            - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
            - output: :math:`(N, C, H, W)`.
        '''
        if self.cfg.task == 'classify':
            images = self.classify_preprocess(im=batch["img"])
            labels = batch["cls"]
        elif self.cfg.task == 'detect':
            raise ValueError('Unsupported attach method!')
            
        
        batch_size = len(images)
        correct = torch.tensor([True] * batch_size)
        target_labels = labels.clone().detach().to(self.device)
        curr_steps = 0
        
        adv_images = []
        for idx in range(batch_size):
            image = images[idx : idx + 1].clone().detach()
            adv_images.append(image)

        while (True in correct) and (curr_steps < steps):
            for idx in range(batch_size):
                if not correct[idx]:
                    continue
                early_stop, pre, adv_image = self._forward_indiv(adv_images[idx], labels[idx])
                adv_images[idx] = adv_image
                target_labels[idx] = pre
                if early_stop:
                    correct[idx] = False
            curr_steps += 1

        adv_images = torch.cat(adv_images).detach()
        return adv_images, target_labels

    def _forward_indiv(self, image, label):
        image.requires_grad = True
        
        if self.cfg.task == 'classify':
            preds = self.model.predict(x=image)
        else:
            raise ValueError('Unsupported attach method!')
        
        fs = preds.to(self.device)
        _, pre = torch.max(fs, dim=-1)
        if pre != label:
            return (True, pre, image)

        ws = self._construct_jacobian(fs, image)
        image = image.detach()

        f_0 = fs[label]
        w_0 = ws[label]

        wrong_classes = [i for i in range(len(fs)) if i != label]
        f_k = fs[wrong_classes]
        w_k = ws[wrong_classes]

        f_prime = f_k - f_0
        w_prime = w_k - w_0
        value = torch.abs(f_prime) / torch.norm(nn.Flatten()(w_prime), p=2, dim=1)
        _, hat_L = torch.min(value, 0)

        delta = (
            torch.abs(f_prime[hat_L])
            * w_prime[hat_L]
            / (torch.norm(w_prime[hat_L], p=2) ** 2)
        )

        target_label = hat_L if hat_L < label else hat_L + 1

        adv_image = image + (1 + overshoot) * delta
        adv_image = torch.clamp(adv_image, min=0, max=1).detach()
        return (False, target_label, adv_image)

    # https://stackoverflow.com/questions/63096122/pytorch-is-it-possible-to-differentiate-a-matrix
    # torch.autograd.functional.jacobian is only for torch >= 1.5.1
    def _construct_jacobian(self, y, x):
        x_grads = []
        for idx, y_element in enumerate(y):
            if x.grad is not None:
                x.grad.zero_()
            y_element.backward(retain_graph=(False or idx + 1 < len(y)))
            x_grads.append(x.grad.clone().detach())
        return torch.stack(x_grads).reshape(*y.shape, *x.shape)
        