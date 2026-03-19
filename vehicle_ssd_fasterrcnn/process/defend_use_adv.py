import torch
from torchvision.models.detection import ssd300_vgg16, fasterrcnn_resnet50_fpn
from torchvision.utils import draw_bounding_boxes
from torchvision import transforms

import os
import glob
from PIL import Image

from sse.sse import sse_print

from defends import Scale, Compression, NeuralCleanse, PGDPurifier, FGSMDenoise

def defend_use_adv(args):
    device = args.device
    
    classes = args.classes
    
    if args.model_name == 'ssd':
        model = ssd300_vgg16(
            weights=None,
            progress=True,
            num_classes=len(classes),
            weights_backbone=None,
            trainable_backbone_layers=None,
            score_thresh=args.score_thresh,
            nms_thresh=args.nms_thresh,
            detections_per_img=args.detections_per_img,
            iou_thresh=args.iou_thresh,
            topk_candidates=args.topk_candidates,
            positive_fraction=args.positive_fraction,
        )
    else:
        model = fasterrcnn_resnet50_fpn(
            weights=None,
            progress=True,
            num_classes=len(classes),
            weights_backbone=None,
            trainable_backbone_layers=None,
            box_score_thresh=args.score_thresh,
            box_nms_thresh=args.nms_thresh,
            box_detections_per_img=args.detections_per_img,
            box_fg_iou_thresh=args.iou_thresh,
            box_bg_iou_thresh=args.iou_thresh,
            box_batch_size_per_image=512,
            box_positive_fraction=args.positive_fraction,
            bbox_reg_weights=None,
        )
    
    event = "model_build"
    data = {
        "status": "success",
        "message": "模型创建完毕.",
        "model_name": args.model_name,
        "class_name": classes
    }
    sse_print(event, data)
    
    model.load_state_dict(torch.load(args.model_path, weights_only=False))
    model.to(device)
    
    event = "model_load"
    data = {
        "status": "success",
        "message": "模型加载完成.",
        "model_name": args.model_name,
        "model_path": args.model_path
    }
    sse_print(event, data)
    
    # ori_images_flod = f"{args.data_path}/ori_images"
    adv_images_flod = f"{args.data_path}/adv_images"
    # print(ori_images_flod)
    # print(adv_images_flod)
    
    # ori_images_paths = glob.glob(os.path.join(ori_images_flod, '*.jpg'))
    adv_images_paths = glob.glob(os.path.join(adv_images_flod, '*.jpg'))
    # ori_images_paths.sort()
    adv_images_paths.sort()
    # print(len(ori_images_paths))
    # print(len(adv_images_paths))
    
    event = "data_load"
    data = {
        "status": "success",
        "message": "对抗样本和原始样本加载完毕.",
        "data_name": args.data_name,
        "samples_number": len(adv_images_paths),
        "data_path": args.data_path
    }
    sse_print(event, data)
    
    try:
        # print(args.defend_method)
        if args.defend_method == 'scale':
            defend = Scale( # 算法输入图像像素为0~255
                            scale=args.scaling_factor,
                            interp=args.interpolate_method
                            )
        elif args.defend_method == 'compression':
            defend = Compression( # 算法输入图像像素为0~255
                                clip_values=(0, 255),
                                quality=args.image_quality,
                                channels_first=False,
                                apply_fit=True,
                                apply_predict=True,
                                verbose=False,
                            )
        elif args.defend_method == 'neural_cleanse':
            defend = NeuralCleanse(kernel_size=args.filter_kernel_size) # 算法输入图像像素为0~255
        elif args.defend_method == 'pgd_purifier':
            defend = PGDPurifier(
                                steps=args.max_iterations, 
                                alpha=args.step_size, 
                                epsilon=args.epsilon*255 # 算法输入图像像素为0~255
                                )
        elif args.defend_method == 'fgsm_denoise':
            defend = FGSMDenoise(epsilon=args.epsilon*255) # 算法输入图像像素为0~255
        else:
            raise ValueError('不支持的防御方法.')

        event = "defend_init"
        data = {
            "status": "success",
            "message": "防御方法初始化完成.",
            "defend_method": args.defend_method
        }
        sse_print(event, data)
    except Exception as e:
        event = "defend_init"
        data = {
            "status": "failure",
            "message": f"{e}",
            "defend_method": args.defend_method
        }
        sse_print(event, data)
        import sys
        sys.exit()



    total_iamges = len(adv_images_paths)
    # ori_pred_conf_sum = 0.0
    # adv_pred_conf_sum = 0.0
    defend_success_count = 0
    defend_failure_count = 0
    
    
    ori_images_flod = f"{args.output_path}/def_images"
    adv_images_flod = f"{args.output_path}/adv_images"
    os.makedirs(ori_images_flod, exist_ok=True)
    os.makedirs(adv_images_flod, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.ToTensor() # 会将像素值范围变为0~1
    ])
    
    model.eval()
    
    for i in range(total_iamges):
        # print(ori_images_paths[i])
        # print(adv_images_paths[i])
        # ori_images = Image.open(ori_images_paths[i])
        adv_images = Image.open(adv_images_paths[i])
        to_tensor = transforms.ToTensor()
        # ori_images = to_tensor(ori_images)
        adv_images = to_tensor(adv_images)
        
        ori_images = adv_images.unsqueeze(0)
        ori_images = (ori_images * 255).to(torch.uint8)
        ori_images, _ = defend(ori_images) # 将防御后的对抗样本作为原始
        # print(ori_images.shape)
        ori_images = ori_images.to(torch.float)
        ori_images = ori_images.squeeze(0)
        
        ori_images = ori_images.to(device)
        adv_images = adv_images.to(device)
        ori_predictions = model(images=[ori_images], targets=None)
        adv_predictions = model(images=[adv_images], targets=None)
        
        # print(len(predictions))
        # print(predictions[0].keys())
        # print(predictions[0]['boxes'].shape)
        # print(predictions[0]['scores'].shape)
        # print(predictions[0]['labels'].shape)
        
        ori_pred_cls = ori_predictions[0]['labels']
        adv_pred_cls = adv_predictions[0]['labels']
        ori_pred_conf = ori_predictions[0]['scores']
        adv_pred_conf = adv_predictions[0]['scores']

        # print(ori_pred_cls)
        # print(ori_pred_conf)
        # print(adv_pred_cls)
        # print(adv_pred_conf)
        
        # actual_class = int(os.path.splitext(os.path.basename(adv_images_paths[i]))[0].split('_')[-1])
        actual_object_number = int(os.path.splitext(os.path.basename(adv_images_paths[i]))[0].split('_')[-1])

        if ori_pred_cls.nelement() == adv_pred_cls.nelement() and ori_pred_cls.nelement() != 0 and (ori_pred_cls == adv_pred_cls).all() and (ori_pred_conf - adv_pred_conf < args.confidence_threshold).all(): # tensor.all()功能: 如果张量tensor中所有元素都是True, 才返回True; 否则返回False
            # defend_result = '失败'
            defend_result = 'failure'
            defend_failure_count += 1
        else:
            # defend_result = '成功'
            defend_result = 'success'
            defend_success_count += 1

        ori_img_path = f"{ori_images_flod}/def_img_{i}_obj_{actual_object_number}_pred_obj_{ori_pred_cls.nelement()}.jpg"
        adv_img_path = f"{adv_images_flod}/adv_img_{i}_obj_{actual_object_number}_pred_obj_{adv_pred_cls.nelement()}.jpg"
        
        ori_pred_labels = [classes[ori_pred_cls[i]] for i in range(ori_pred_cls.nelement())]
        adv_pred_labels = [classes[adv_pred_cls[i]] for i in range(adv_pred_cls.nelement())]
        ori_pred_labels_scores = [label+f" {ori_pred_conf[i]:.2f}" for i, label in enumerate(ori_pred_labels)]
        adv_pred_labels_scores = [label+f" {adv_pred_conf[i]:.2f}" for i, label in enumerate(adv_pred_labels)]
        ori_images = draw_bounding_boxes(
                                image=ori_images, 
                                boxes=ori_predictions[0]["boxes"],
                                labels=ori_pred_labels_scores,
                                colors="red",
                                width=1,
                                font_size=32)
        adv_images = draw_bounding_boxes(
                                image=adv_images, 
                                boxes=adv_predictions[0]["boxes"],
                                labels=adv_pred_labels_scores,
                                colors="red",
                                width=1,
                                font_size=32)
        
        
        to_pil = transforms.ToPILImage()
        pil_image = to_pil(ori_images)
        pil_image.save(ori_img_path)
        pil_image = to_pil(adv_images)
        pil_image.save(adv_img_path)
        
        
        event = "defend"
        data = {
            "message": "正在执行防御...",
            "progress": int(i/total_iamges*100),
            # "log": f"[{int(i/total_iamges*100)}%] 正在执行攻击... 原始样本: {ori_img_path}, 原始样本预测准确率: {ori_pred_conf*100:.2f}%, 原始样本预测类别: {ori_pred_cls.item()}, 对抗样本: {adv_img_path}, 对抗样本预测准确率: {adv_pred_conf*100:.2f}%, 对抗样本预测类别: {ori_pred_cls.item()}, 原始样本实际类别: {labels[i].item()}, 攻击结果: {defend_result}"
            "log": f"[{int(i/total_iamges*100)}%] 正在执行防御...",
            "details": {
                "defend_sample": {
                    # "confidence": f"{ori_pred_conf*100:.2f}%",
                    # "predict_class": ori_pred_cls,
                    "object_number": ori_pred_cls.nelement(),
                    "predict_class": ori_pred_labels,
                    "confidence": ori_pred_conf.tolist(),
                    "file_path": ori_img_path
                },
                "adversarial_sample": {
                    # "confidence": f"{adv_pred_conf*100:.2f}%",
                    # "predict_class": adv_pred_cls,
                    "object_number": adv_pred_cls.nelement(),
                    "predict_class": adv_pred_labels,
                    "confidence": adv_pred_conf.tolist(),
                    "file_path": adv_img_path
                },
                # "actual_class": actual_class,
                "actual_object_number": actual_object_number,
                "defend_result": defend_result
            }
        }
        sse_print(event, data)
        
    
    event = "final_result"
    data = {
        "message": "防御执行完成.",
        "progress": 100,
        "log": f"[100%] 防御执行完成.",
        "details": {
            "defend_samples": ori_images_flod,
            "adversarial_samples": adv_images_flod,
            "summary": {
                "total_iamges": total_iamges,
                "task_success_count": defend_success_count,
                "task_failure_count": defend_failure_count,
                # "defend_samples_pred_confidence": f"{ori_pred_conf_sum/total_iamges*100:.2f}%",
                # "adversarial_samples_pred_confidence": f"{adv_pred_conf_sum/total_iamges*100:.2f}%"
            }
        }
    }
    sse_print(event, data)
        
