import json
import zipfile
import os
import sys
import glob
from pathlib import Path
import numpy as np

def sse_print(event: str, data: dict) -> str:
    """
    SSE 打印
    :param event: 事件名称
    :param data: 事件数据（字典或能被 json 序列化的对象）
    :return: SSE 格式字符串
    """
    # 将数据转成 JSON 字符串
    json_str = json.dumps(data, ensure_ascii=False, default=lambda obj: obj.item() if isinstance(obj, np.generic) else obj)
    
    # 按 SSE 协议格式拼接
    message = f"event: {event}\n" \
              f"data: {json_str}\n"
    print(message)



# def sse_input_path_validated(args):
#     if not os.path.exists(args.input_path):
#         event = "input_path_validated"
#         data = {
#             "status": "failure",
#             "message": "Input path not found."
#         }
#         sse_print(event, data)
#         raise ValueError('Input path not found.')
#     else:
#         event = "input_path_validated"
#         data = {
#             "status": "success",
#             "message": "Input path is valid and complete.",
#             "file_name": args.input_path
#         }
#         sse_print(event, data)
        
#         if not os.path.exists(f'{args.input_path}/data'):
#             event = "input_data_validated"
#             data = {
#                 "status": "failure",
#                 "message": "Input data file not found."
#             }
#             sse_print(event, data)
#             raise ValueError('Input data file not found.')
#         else:
#             event = "input_data_validated"
#             data = {
#                 "status": "success",
#                 "message": "Input data file is valid and complete.",
#                 "file_name": glob.glob(os.path.join(f'{args.input_path}/data', '*/'))[0]
#             }
#             sse_print(event, data)
            
#         if not os.path.exists(f'{args.input_path}/model'):
#             event = "input_model_validated"
#             data = {
#                 "status": "failure",
#                 "message": "Input model file not found."
#             }
#             sse_print(event, data)
#             raise ValueError('Input model file not found.')
#         else:
#             event = "input_model_validated"
#             data = {
#                 "status": "success",
#                 "message": "Input model file is valid and complete.",
#                 "file_name": glob.glob(os.path.join(f'{args.input_path}/model', '*'))[0]
#             }
#             sse_print(event, data)


def sse_input_path_validated(args):
    try:
        if os.path.exists(args.input_path):
            event = "input_path_validated"
            data = {
                "status": "success",
                "message": "Input path is valid and complete.",
                "file_name": args.input_path
            }
            sse_print(event, data)
            
            try:
                if os.path.exists(f'{args.input_path}/data'):
                    event = "input_data_validated"
                    data = {
                        "status": "success",
                        "message": "Input data file is valid and complete.",
                        "file_name": glob.glob(os.path.join(f'{args.input_path}/data', '*/'))[0]
                    }
                    sse_print(event, data)
                else:
                    raise ValueError('Input data file not found.')
            except Exception as e:
                event = "input_data_validated"
                data = {
                    "status": "failure",
                    "message": f"{e}"
                }
                sse_print(event, data)
                
            try:
                if os.path.exists(f'{args.input_path}/model'):
                    event = "input_model_validated"
                    data = {
                        "status": "success",
                        "message": "Input model file is valid and complete.",
                        "file_name": glob.glob(os.path.join(f'{args.input_path}/model', '*'))[0]
                    }
                    sse_print(event, data)
                else:
                    raise ValueError('Input model file not found.')
            except Exception as e:
                event = "input_model_validated"
                data = {
                    "status": "failure",
                    "message": f"{e}"
                }
                sse_print(event, data)
        else:
            raise ValueError('Input path not found.')
    except Exception as e:
        event = "input_path_validated"
        data = {
            "status": "failure",
            "message": f"{e}"
        }
        sse_print(event, data)
        
    


def sse_output_path_validated(args):
    try:
        if os.path.exists(args.output_path):
            event = "output_path_validated"
            data = {
                "status": "success",
                "message": "Output path is valid and complete.",
                "file_name": args.output_path
            }
            sse_print(event, data)
        else:
            raise ValueError('Output path not found.')
    except Exception as e:
        event = "output_path_validated"
        data = {
            "status": "failure",
            "message": f"{e}"
        }
        sse_print(event, data)
            
        

def sse_adv_samples_gen_validated(adv_image_name):
        event = "adv_samples_gen_validated"
        data = {
            "status": "success",
            "message": "adversarial sample is generated.",
            "file_name": adv_image_name
        }
        sse_print(event, data)


def sse_clean_samples_gen_validated(clean_image_name):
        event = "clean_samples_gen_validated"
        data = {
            "status": "success",
            "message": "clean sample is generated.",
            "file_name": clean_image_name
        }
        sse_print(event, data)
        
# def sse_input_validated(input_path):
#     if not os.path.exists(input_path):
#         event = "input_validated"
#         data = {
#             "status": "failure",
#             "message": "Input path not found."
#         }
#         sse_print(event, data)
#         raise ValueError('Input path not found.')
#     elif not input_path.endswith('.zip'):
#         event = "input_validated"
#         data = {
#             "status": "failure",
#             "message": "Input is not zip file."
#         }
#         sse_print(event, data)
#     else:
#         event = "input_validated"
#         data = {
#             "status": "success",
#             "message": "Input zip file is valid and complete.",
#             "file_name": os.path.basename(input_path),
#             "file_size": f"{os.path.getsize(input_path)/1024/1024:.2f}MB"
#         }
#         sse_print(event, data)


def sse_working_path_created(working_path):
    if not os.path.exists(working_path):
        os.makedirs(working_path)
        event = "working_path_created"
        data = {
            "status": "success",
            "message": "Temporary working directory created.",
            "directory_path": working_path
        }
        sse_print(event, data)


def sse_source_unzip_completed(dataset_path, working_path):
    
    # os.makedirs(dataset_working_path, exist_ok=True)
    
    with zipfile.ZipFile(dataset_path, 'r') as zipf:
        zipf.extractall(working_path)
        
        dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
        dataset_working_path = os.path.join(working_path, dataset_name)
        path = Path(dataset_working_path)
        image_files = list(path.rglob("*.jpg")) + list(path.rglob("*.JPG")) + list(path.rglob("*.jpeg")) + list(path.rglob("*.JPEG")) + list(path.rglob("*.png")) + list(path.rglob("*.PNG"))
        image_count = len(image_files)
        
        event = "source_unzip_completed"
        data = {
            "status": "success",
            "message": "Source zip file extracted successfully.",
            "image_count": image_count,
            "output_directory": dataset_working_path
        }
        sse_print(event, data)

def sse_model_loaded(model_name, weight_path):
    event = "model_loaded"
    data = {
        "status": "success",
        "message": "Model loaded successfully.",
        "model_name": model_name,
        "weight_path": weight_path
    }
    sse_print(event, data)

    