import argparse
from easydict import EasyDict
import os
import glob

from sse.sse import sse_input_path_validated, sse_output_path_validated

from torchvision.models._meta import _COCO_CATEGORIES
from torchvision.models._meta import _VOC_CATEGORIES

def parse_args():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='./input', help='input path')
    parser.add_argument('--output_path', type=str, default='./output', help='output path')
    
    parser.add_argument('--process', type=str, default='train', choices=['adv', 'attack', 'defend', 'train', 'test', 'predict', 'sample'], help='process name')
    # parser.add_argument('--model', type=str, default='drone_yolo', choices=['drone_yolo'], help='model name')
    # parser.add_argument('--data', type=str, default='yolo_drone_detection', choices=['yolo_drone_detection'], help='data name')
    # parser.add_argument('--class_number', type=int, default=1, choices=[1, 1000], help='number of class. 1 for yolo_drone_detection, 1000 for imagenet10')
    
    parser.add_argument('--attack_method', type=str, default='fgsm', choices=['fgsm', 'mifgsm', 'vmifgsm', 'pgd', 'bim', 'cw', 'deepfool', 'gn', 'jitter'], help='attack method')
    parser.add_argument('--defend_method', type=str, default='scale', choices=['scale', 'compression', 'fgsm_denoise', 'neural_cleanse', 'pgd_purifier', 'adversarial_training'], help='defend method')
    
    parser.add_argument('--confidence_threshold', type=float, default=0.1, help='confidence threshold')
    parser.add_argument('--resume_from_checkpoint', type=bool, default=False, help='resume from checkpoint for train or fit')
    
    parser.add_argument('--selected_samples', type=int, default=64, help='the number of generated adversarial sample for attack method')
    parser.add_argument('--adversarial_sample_proportion', type=int, default=50, help='adversarial sample proportion in dataset when defend train for defend method')
    
    parser.add_argument('--epochs', type=int, default=100, help='epochs')
    parser.add_argument('--batch', type=int, default=128, help='batch size')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='device')
    parser.add_argument('--workers', type=int, default=16, help='dataloader workers (per RANK if DDP)')
    
    parser.add_argument('--classes', type=list, default=_COCO_CATEGORIES, choices=[_COCO_CATEGORIES, _VOC_CATEGORIES], help='dataset classes list')
    parser.add_argument('--score_thresh', type=float, default=0.3, help='score threshold used for postprocessing the detections')
    parser.add_argument('--nms_thresh', type=float, default=0.1, help='NMS threshold used for postprocessing the detections')
    parser.add_argument('--iou_thresh', type=float, default=0.7, help='minimum IoU between the anchor and the GT box so that they can be considered as positive during training')
    parser.add_argument('--detections_per_img', type=int, default=5, help='number of best detections to keep after NMS')
    parser.add_argument('--topk_candidates', type=int, default=10, help='number of best detections to keep before NMS')
    parser.add_argument('--positive_fraction', type=float, default=0.5, help='a number between 0 and 1 which indicates the proportion of positive proposals used during the training of the classification head. It is used to estimate the negative to positive ratio')
    
    parser.add_argument('--epsilon', type=float, default=15/255, help='epsilon for attack method and defend medthod')
    parser.add_argument('--step_size', type=float, default=2/255, help='epsilon for attack method and defend medthod')
    parser.add_argument('--max_iterations', type=int, default=10, help='epsilon for attack method and defend medthod')
    
    parser.add_argument('--decay', type=float, default=1.0, help='momentum factor of mifgsm for attack method')
    parser.add_argument('--sampled_examples', type=int, default=5, help='the number of sampled examples in the neighborhood of vmifgsm for attack method')
    
    parser.add_argument('--random_start', type=bool, default=True, help='initial random start for attack method')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate of optimization for attack method')
    
    parser.add_argument('--std', type=float, default=0.1, help='standard deviation for gn and jitter attack method')
    parser.add_argument('--scale', type=int, default=10, help='scale for jitter attack method')
    
    parser.add_argument('--scaling_factor', type=float, default=0.5, help='scaling factor (0, 1) for defend method')
    parser.add_argument('--interpolate_method', type=str, default='bilinear', choices=['bilinear', 'nearest'], help='interpolate method for defend method')
    parser.add_argument('--image_quality', type=int, default=90, help='the image quality for defend method, on a scale from 1 (worst) to 95 (best). Values above 95 should be avoided.')
    parser.add_argument('--filter_kernel_size', type=int, default=3, help='filter kernel size for defend method.')
    
    args = parser.parse_args()
    args_dict = vars(args)
    args_dict_environ = {}
    for key, value in args_dict.items():
        if key in ['input_path', 'output_path']:
            args_dict_environ[key] = type_switch(os.getenv(key.upper(), value), value)
        else:
            args_dict_environ[key] = type_switch(os.getenv(key, value), value)
    args_easydict = EasyDict(args_dict_environ)
    return args_easydict
    
def type_switch(environ_value, value):
    if isinstance(value, bool):
        return bool(environ_value)
    elif isinstance(value, int):
        return int(environ_value)
    elif isinstance(value, float):
        return float(environ_value)
    elif isinstance(value, str):
        return environ_value
    elif isinstance(value, list):
        return environ_value
    
def add_args(args):
    model_yaml = glob.glob(os.path.join(os.path.join(f'{args.input_path}/model', '*/'), '*.yaml'))[0]
    # print(model_yaml)
    args.model_yaml = model_yaml
    model_name = os.path.splitext(os.path.basename(model_yaml))[0]
    # print(model_name)
    args.model_name = model_name
    if args.process != 'train' or (args.process == 'train' and args.resume_from_checkpoint):
        model_path = glob.glob(os.path.join(os.path.join(f'{args.input_path}/model', '*/'), '*.pth'))[0]
        # print(model_path)
        args.model_path = model_path
    
    
    if args.process == 'attack' or (args.process == 'defend' and args.defend_method != 'adversarial_training') or args.process == 'predict':
        data_path = f'{args.input_path}/data'
        
        data_yaml = glob.glob(os.path.join(f'{args.input_path}/data', '*.yaml'))[0]
        # print(data_yaml)
        args.data_yaml = data_yaml
        data_name = os.path.splitext(os.path.basename(data_yaml))[0]
        # print(data_name)
        args.data_name = data_name
    
    else:
        # data_path = glob.glob(os.path.join(os.path.join(f'{args.input_path}/data', '*/'), '*/'))[0]
        data_path = glob.glob(os.path.join(f'{args.input_path}/data', '*/'))[0]
        
        data_yaml = glob.glob(os.path.join(os.path.join(f'{args.input_path}/data', '*/'), '*.yaml'))[0]
        # print(data_yaml)
        args.data_yaml = data_yaml
        data_name = os.path.splitext(os.path.basename(data_yaml))[0]
        # print(data_name)
        args.data_name = data_name
    
    # print(data_path)
    args.data_path = data_path
    
    return args


def main(args):
    args = add_args(args)
    
    if args.process == 'adv':
        from process.adv import adv
        args.batch = 1
        adv(args)
    elif args.process == 'attack':
        from process.attack_use_adv import attack_use_adv
        args.batch = 1
        attack_use_adv(args)
    elif args.process == 'defend':
        if args.defend_method == 'adversarial_training':
            from process.adv_train import adv_train
            adv_train(args)
        else:
            from process.defend_use_adv import defend_use_adv
            args.batch = 1
            defend_use_adv(args)
    elif args.process == 'train':
        from process.train import train
        train(args)
    elif args.process == 'test':
        from process.test import test
        test(args)
    elif args.process == 'predict':
        from process.predict import predict
        args.batch = 1
        predict(args)
    elif args.process == 'sample': 
        from process.sample import sample
        sampled_data_path = f'{args.output_path}/sampled_' + args.data_name
        sample(src_dir=args.data_path, dst_dir=sampled_data_path, train_sample_num=args.selected_samples, val_sample_num=args.selected_samples)
        os.system(f"cp {args.data_yaml} {args.output_path}")
        from sse.sse import sse_print
        event = "final_result"
        data = {
            "message": "数据集抽取完毕.",
            "progress": 100,
            "log": f"[100%] 从{args.data_path}数据集中抽取{args.selected_samples}张样本生成小数据集保存在{sampled_data_path}"
        }
        sse_print(event, data)
    else:
        raise ValueError('任务不支持.')
 
        
if __name__ == '__main__':
    args = parse_args()
    
    sse_input_path_validated(args)
    sse_output_path_validated(args)
    main(args)
    
    
