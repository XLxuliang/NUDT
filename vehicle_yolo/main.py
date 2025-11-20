import argparse
import yaml
from easydict import EasyDict
import os
import glob

from utils.sse import sse_input_path_validated, sse_output_path_validated
from utils.yaml_rw import load_yaml, save_yaml
from nudt_ultralytics.main import main as yolo

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='./input', help='input path')
    parser.add_argument('--output_path', type=str, default='./output', help='output path')
    
    parser.add_argument('--process', type=str, default='defend', choices=['adv', 'attack', 'defend', 'train'], help='process name')
    parser.add_argument('--model', type=str, default='yolov8', choices=['yolov5', 'yolov8', 'yolov10'], help='model name')
    parser.add_argument('--data', type=str, default='imagenet10', choices=['coco8', 'imagenet10'], help='data name')
    parser.add_argument('--task', type=str, default='classify', choices=['detect', 'classify'], help='task name. detect for coco8, classify for imagenet10')
    parser.add_argument('--class_number', type=int, default=1000, choices=[80, 1000], help='number of class. 80 for coco8, 1000 for imagenet10')
    
    parser.add_argument('--attack_method', type=str, default='pgd', choices=['cw', 'deepfool', 'bim', 'fgsm', 'pgd'], help='attack method')
    parser.add_argument('--defend_method', type=str, default='scale', choices=['scale', 'compression', 'fgsm_denoise', 'neural_cleanse', 'pgd_purifier'], help='defend method')
    
    parser.add_argument('--cfg_path', type=str, default='./cfgs', help='cfg path')
    
    parser.add_argument('--epochs', type=int, default=100, help='epochs')
    parser.add_argument('--batch', type=int, default=8, help='batch size')
    # parser.add_argument('--device', type=int, default=0, help='which gpu for cuda')
    parser.add_argument('--device', type=str, default='cpu', help='which gpu for cuda')
    parser.add_argument('--workers', type=int, default=0, help='dataloader workers (per RANK if DDP)')
    
    parser.add_argument('--gen_adv_sample_num', type=int, default=10, help='the number of generated adversarial sample for attack method')
    parser.add_argument('--epsilon', type=float, default=8/255, help='epsilon for attack method')
    parser.add_argument('--step_size', type=float, default=2/255, help='epsilon for attack method')
    parser.add_argument('--max_iterations', type=int, default=50, help='epsilon for attack method')
    parser.add_argument('--random_start', type=bool, default=False, help='initial random start for attack method')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate of optimization for attack method')
    parser.add_argument('--loss_function', type=str, default='cross_entropy', choices=['cross_entropy', 'mse', 'l1', 'binary_cross_entropy'], help='loss function for attack method')
    parser.add_argument('--optimization_method', type=str, default='adam', choices=['adam', 'sgd'], help='optimization for attack method')
    
    args = parser.parse_args()
    args_dict = vars(args)
    args_dict_environ = {}
    for key, value in args_dict.items():
        args_dict_environ[key] = type_switch(os.getenv(key.upper(), value), value)
    args_easydict = EasyDict(args_dict_environ)
    return args_easydict


def type_switch(environ_value, value):
    if isinstance(value, int):
        return int(environ_value)
    elif isinstance(value, float):
        return float(environ_value)
    elif isinstance(value, bool):
        return bool(environ_value)
    elif isinstance(value, str):
        return environ_value
    
def yolo_cfg(args):
    
    model_yaml = f'./nudt_ultralytics/cfgs/models/{args.task}/{args.model}.yaml'
    model_cfg = load_yaml(model_yaml)
    model_cfg['nc'] = args.class_number
    model_yaml = f'{args.cfg_path}/{args.model}.yaml'
    save_yaml(model_cfg, model_yaml)
    
    data_yaml = f'./nudt_ultralytics/cfgs/datasets/{args.data}.yaml'
    data_cfg = load_yaml(data_yaml)
    # data_cfg['path'] = f'{args.input_path}/data/{args.data}' # 数据集名称与数据集文件夹名称绑定成一样
    data_cfg['path'] = glob.glob(os.path.join(f'{args.input_path}/data', '*/'))[0] # input_path/data目录下有且只有一个数据集文件夹
    data_yaml = f'{args.cfg_path}/{args.data}.yaml'
    save_yaml(data_cfg, data_yaml)
    
    cfg_yaml = f'./nudt_ultralytics/cfgs/models/{args.task}/default.yaml'
    cfg = load_yaml(cfg_yaml)
    cfg = EasyDict(cfg)
    cfg.task = args.task
    cfg.model = f'{args.cfg_path}/{args.model}.yaml'
    if args.task == 'classify':
        cfg.data = data_cfg['path']
    else:
        cfg.data = data_yaml
    cfg.save_dir = args.output_path
    cfg.project = args.model
    cfg.name = args.process
    if args.process == 'adv':
        cfg.mode = 'predict'
        cfg.batch = 1
        # cfg.pretrained = f'{args.input_path}/model/{args.model}.pt' # 模型名称与模型权重文件名称绑定成一样
        cfg.pretrained = glob.glob(os.path.join(f'{args.input_path}/model', '*'))[0] # input_path/model目录下有且只有一个权重文件
        cfg.device = args.device
    elif args.process == 'attack':
        cfg.mode = 'validate'
        cfg.batch = 1
        # cfg.pretrained = f'{args.input_path}/model/{args.model}.pt' # 模型名称与模型权重文件名称绑定成一样
        cfg.pretrained = glob.glob(os.path.join(f'{args.input_path}/model', '*'))[0] # input_path/model目录下有且只有一个权重文件
        cfg.device = args.device
        cfg.attack_or_defend = 'attack'
        cfg.attack_method = args.attack_method
    elif args.process == 'defend':
        cfg.mode = 'predict'
        # cfg.batch = 1
        # cfg.pretrained = f'{args.input_path}/model/{args.model}.pt' # 模型名称与模型权重文件名称绑定成一样
        cfg.pretrained = glob.glob(os.path.join(f'{args.input_path}/model', '*'))[0] # input_path/model目录下有且只有一个权重文件
        cfg.attack_or_defend = 'defend'
        cfg.defend_method = args.defend_method
    elif args.process == 'train':
        cfg.mode = 'train'
        cfg.epochs = args.epochs
        cfg.batch = args.batch
        cfg.device = args.device
    
    cfg = dict(cfg)
    args.cfg_yaml = f'{args.cfg_path}/default.yaml'
    save_yaml(cfg, args.cfg_yaml)
    
    return args

def main(args):
    args = yolo_cfg(args)
    yolo(args)
        
if __name__ == '__main__':
    args = parse_args()
    
    sse_input_path_validated(args)
    sse_output_path_validated(args)
    # sse_working_path_created(args.working_path)
    # sse_source_unzip_completed(args.dataset_path, args.working_path)
    main(args)
    
    
    
    
    
    
