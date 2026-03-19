# 车辆识别场景

## 概述
本项目基于 [torchvision](https://docs.pytorch.org/vision/stable/index.html) 库实现车辆识别场景下ssd, faster-rcnn的车辆识别任务（只支持使用VOC数据集）。它支持对抗样本生成、攻击评估、防御机制和模型训练。

## 环境变量：
* `input_path`（str 必填）: 指定输入路径，在此路径下有权重文件和数据集文件。
* `output_path`（str 必填）: 指定输出路径，在此路径下保存生成的对抗样本和防御训练的权重。
* `process`（str 必填）: 指定进程名称，支持枚举值:`adv`, `attack`, `defend`, `train`, `test`, `predict`, `sample`。
* ~~`model`（str 必填）: 指定模型名称，支持枚举值:`yolov5`, `yolov8`, `yolov10` 。~~
* ~~`data`（str 必填）: 指定数据集，支持枚举值:`kitti`, `bdd100k`, `ua-detrac`, `dawn`, `special_vehicle`, `flir_adas`。~~
* ~~`class_number`（int 必填）: 指定目标类别数量，与数据集绑定，对于kitti数据集为`8` 。~~
* `attack_method`（str 选填，默认为`fgsm`）: 指定攻击方法，若`process`为`adv`或`attack`则必填，支持枚举值（第一个为默认值）: `fgsm`, `mifgsm`, `vmifgsm`, `pgd`, `bim`, `cw`, `deepfool`, `gn`, `jitter`。
* `defend_method`（str 选填，默认为`scale`）: 指定防御方法，若`process`为`defend`则必填，支持枚举值（第一个为默认值）:`scale`, `compression`, `fgsm_denoise`, `neural_cleanse`, `pgd_purifier`, `adversarial_training`。
* `selected_samples`（int 选填，默认为64，0表示数据集全部样本数据）: 若`process`为`adv`时有效，生成对抗样本时使用的样本数。
* `confidence_threshold`（float 选填，默认为`0.1`）：攻击防御之后置信度变化阈值，超过该值说明攻击防御成功，若`process`为`attack`或`defend`时有效。
* `resume_from_checkpoint`（bool 选填，默认为`False`）：是否在训练时在已有的权重基础上进行训练，若`process`为`train`时有效，支持枚举值（第一个为默认值）:`False`, `True`。
* `adversarial_sample_proportion`（int 选填，默认为50）: 若`process`为`defend`时有效，防御训练时，训练数据集中对抗样本所占百分比，取值在（0，100）之间。
* `epochs`（int 选填，默认为`100`）：训练迭代次数，若`process`为`train`时有效。
* `batch`（int 选填，默认为`16`）：训练批处理大小，若`process`为`train`或`test`时有效。
* `device`（str 选填，默认为`cpu`）：使用cpu或cuda，支持枚举值（第一个为默认值）:`cpu`, `cuda`。
* `workers`（int 选填，默认为`0`）：加载数据集时workers的数量。
* `epsilon`（float 选填，默认为`8/255`）：扰动强度参数，控制对抗扰动大小。
* `step_size`（float 选填，默认为`2/255`）：步长，迭代攻击的更新幅度。
* `max_iterations`（int 选填，默认为`50`）：最大迭代次数。
* `decay`（float 选填，默认为`1.0`）：延时，取值在（0，1）之间，其值越小，迭代轮数靠前算出来的梯度对当前的梯度方向影响越小。
* `sampled_examples`（int 选填，默认为`5`）：邻近抽取的样例数。
* `random_start`（bool 选填，默认为`False`）：是否随机初始化扰动，支持枚举值（第一个为默认值）:`False`, `True`。
* ~~`loss_function`（str 选填，默认为`cross_entropy`）：损失函数类型，支持枚举值（第一个为默认值）:`cross_entropy`, `mse`, `l1`, `binary_cross_entropy`。~~
* ~~`optimization_method`（str 选填，默认为`adam`）：优化方法，支持枚举值（第一个为默认值）:`adam`, `sgd`。~~
* `lr`（float 选填，默认为`0.001`）：优化器学习率。
* `scale`（int 选填，默认为`10`）：尺度。
* `std`（float 选填，默认为`0.1`）：标准差。
* `scaling_factor`（float 选填，默认为`0.9`）：缩放因子。
* `interpolate_method`（str 选填，默认为`bilinear`）：插值方法，支持枚举值（第一个为默认值）:`bilinear`, `nearest`。
* `image_quality`（int 选填，默认为`90`）：图像质量。
* `filter_kernel_size`（int 选填，默认为`3`）：滤波器核大小。


### 攻击防御方法的有效参数
| 参数 | fgsm | mifgsm | vmifgsm | pgd | bim | cw | deepfool | gn | jitter | scale | compression | neural_cleanse | pgd_purifier | fgsm_denoise | adversarial_training |
|------|------|--------|---------|-----|-----|----|----------|----|--------|-------|-------------|----------------|--------------|--------------|--------------|
| epsilon | 1 | 1 | 1 | 1 | 1 | | | | 1 | | | | 1 | 1 | |
| step_size | | 1 | 1 | 1 | 1 | | | | 1 | | | | 1 | | |
| max_iterations | | 1 | 1 | 1 | 1 | 1 | 1 | | 1 | | | | 1 | |
| decay | | 1 | 1 | | | | | | | | | | | | |
| sampled_examples | | | 1 | | | | | | | | | | | | |
| random_start | | | | 1 | | | | | 1 | | | | | | |
| lr | | | | | | 1 | | | | | | | | | |
| std | | | | | | | | 1 | 1 | | | | | | |
| scale | | | | | | | | | 1 | | | | | | |
| scaling_factor | | | | | | | | | | 1 | | | | | |
| interpolate_method | | | | | | | | | | 1 | | | | | |
| image_quality | | | | | | | | | | | 1 | | | | |
| filter_kernel_size | | | | | | | | | | | | 1 | | | |


### 说明
- `process`为`adv`：生成对抗样本：使用原始数据集的val子集 -> 通过`selected_samples`参数从原始数据集中选择多少张样本 -> 选择攻击方法（需要使用模型） -> 生成对抗样本 -> 保存对抗样本数据集（原始样本也会一并保存） \
- `process`为`attack`：攻击：使用生成对抗样本数据集 -> 加载模型进行推理 -> 通过比较原始样本和对抗样本的预测类别是否相同和置信度降低幅度是否大于`confidence_threshold`判断是否攻击成功 -> 打印成功率 -> 保存识别图片 \
- `process`为`defend`：防御：使用生成对抗样本数据集 -> 选择防御方法 -> 加载模型进行推理 -> 通过比较对抗样本经过防御方法后的预测类别是否正确和置信度提升幅度是否大于 `confidence_threshold`判断是否防御成功 -> 打印成功率 -> 保存识别图片 \
- `process`为`train`：训练：使用原始数据集的train子集 -> 通过`resume_from_checkpoint`参数决定是否在已有的权重基础上进行训练 -> 执行训练 -> 保存权重文件 \
- `process`为`test`：测试：使用原始数据集的val子集 -> 执行测试 -> 保存预测图片 \
- `process`为`predict`：预测：使用图片进行推理（所有图片放在一个文件夹中，该文件夹与一个.yaml一起压缩） -> 保存运行图像 \
- `process`为`sample`：抽样：使用原始数据集 -> 通过`selected_samples`参数从原始数据集中选择多少张样本（train和val子集都选择） -> 按原始数据集格式保存

### 调参
1. score_thresh (float): 用于后处理检测结果的分数阈值。
    - 作用：在模型输出的大量检测框中，根据每个框的置信度分数（表示该框包含目标的可能性）进行筛选，只有分数高于此阈值的检测框才会被保留。
    - 影响：
        - 增大：阈值提高，只有置信度更高的检测框被保留，因此误检（假阳性）会减少，但可能会漏掉一些置信度较低的真实目标（假阴性增加）。
        - 减小：阈值降低，更多的检测框被保留，包括一些置信度较低的，因此可能检测到更多真实目标（假阴性减少），但误检也会增加（假阳性增加）。
2. nms_thresh (float): 非极大值抑制（NMS）的阈值。
    - 作用：在目标检测中，同一个目标可能会被多个检测框检测到，NMS用于去除这些重叠的框。具体来说，对于同一类别的目标，根据置信度排序，然后抑制那些与最高置信度框的IoU（交并比）超过阈值的框。
    - 影响：
        - 增大：阈值提高，意味着更宽松的抑制，允许保留更多重叠的检测框，可能会导致同一个目标被多个框检测到。
        - 减小：阈值降低，抑制更严格，同一个目标只保留一个或少数几个最置信的框，但可能会错误地抑制一些定位略有不同但可能是不同目标的框。
3. detections_per_img (int): 经过NMS后，每张图片保留的最佳检测数量。
    - 作用：限制每张图片最终输出的检测框数量，通常按照置信度排序，保留前N个。
    - 影响：
        - 增大：保留更多的检测框，可能提高召回率，但也会增加计算负担和可能的误检。
        - 减小：保留更少的检测框，可能会漏检一些目标，但输出更简洁，计算负担小。
4. iou_thresh (float): 训练时用于确定正样本的IoU阈值。
    - 作用：在训练阶段，我们需要为每个锚框（anchor）分配标签。这个参数定义了锚框与真实框（GT）之间的最小IoU，当IoU大于此阈值时，该锚框被视为正样本（包含目标）。
    - 影响：
        - 增大：阈值提高，只有与真实框重叠程度更高的锚框才被视为正样本，这些正样本的质量可能更高，但数量会减少，可能导致训练样本不足。
        - 减小：阈值降低，更多的锚框被视为正样本，增加了训练样本的数量，但可能会引入一些质量较低的正样本（与真实框重叠不高），导致训练困难。
5. topk_candidates (int): 在NMS之前，保留的候选检测框数量。
    - 作用：在NMS之前，我们通常不会对所有的候选框（可能成千上万）都进行NMS，而是先按置信度排序，保留前k个，然后再进行NMS。这个参数就是控制这个k值。
    - 影响：
        - 增大：保留更多的候选框进行NMS，可能会提高召回率，但增加计算量。
        - 减小：保留更少的候选框，计算量减少，但可能会漏掉一些低置信度但却是真实目标的框。
6. positive_fraction (float): 训练分类头时，正样本所占的比例。
    - 作用：在训练分类头时，我们需要平衡正负样本。这个参数定义了正样本在总样本中的比例。例如，如果设置为0.25，则正样本占25%，负样本占75%。
    - 影响：
        - 增大：提高正样本比例，可能会使模型对正样本（目标）的学习更充分，但可能会减少负样本（背景）的学习，导致误检增加。
        - 减小：降低正样本比例，增加负样本比例，可能帮助模型更好地区分背景，但可能会降低对目标的学习能力。

注意：这些参数需要根据具体任务和数据集进行调整，以达到最佳性能。通常，这些参数在验证集上进行调优。


## 快速开始

### 构建 Docker 镜像
```bash
cd vehicle_ssd_fasterrcnn
docker build -t vehicle_ssd_fasterrcnn:latest .
```

### 输入输出目录结构
```
docker_inout_dir/
├── input/
│   ├── model/
│   │   └── model_name/                 # 模型目录
│   │       └── weight.pt               # 模型权重
│   │       └── model_cfg.yaml          # 模型配置文件
│   └── data/
│       └── data_name/                  # 数据集目录
│           └── data/                   # 数据集
│           └── data_cfg.yaml           # 数据集配置文件
├── output/
```

### 运行 Docker 镜像
```bash
cd docker_inout_dir

docker run --rm \
    -v ./input:/project/input:ro \
    -v ./output:/project/output:rw \
    -e INPUT_PATH=/project/input \
    -e OUTPUT_PATH=/project/output \
    -e process=train \
    -e attack_method=fgsm \
    -e defend_method=scale \
    -e epochs=100 \
    -e batch=16 \
    -e device=cuda \
    -e workers=0 \
    -e selected_samples=64 \
    -e epsilon=0.0313 \
    -e step_size=0.0078 \
    -e max_iterations=50 \
    -e decay=1.0 \
    -e sampled_examples=5 \
    -e random_start=False \
    -e loss_function=cross_entropy \
    -e optimization_method=adam \
    -e lr=0.001 \
    -e scaling_factor=0.9 \
    -e interpolate_method=bilinear \
    -e image_quality=90 \
    -e filter_kernel_size=3 \
    vehicle_ssd_fasterrcnn:latest
```


## 资料
### [模型](https://docs.pytorch.org/vision/stable/models.html):`https://docs.pytorch.org/vision/stable/models.html`
- [SSD](https://docs.pytorch.org/vision/stable/models/ssd.html):`https://docs.pytorch.org/vision/stable/models/ssd.html`

    - [ssd300_vgg16](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.ssd300_vgg16.html#torchvision.models.detection.ssd300_vgg16):`https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.ssd300_vgg16.html#torchvision.models.detection.ssd300_vgg16`

    - 代码: torchvision.models.detection.ssd.py, 其中有权重下载链接

- [Faster R-CNN](https://docs.pytorch.org/vision/stable/models/faster_rcnn.html):`https://docs.pytorch.org/vision/stable/models/faster_rcnn.html`

    - [fasterrcnn_resnet50_fpn](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html#torchvision.models.detection.fasterrcnn_resnet50_fpn):`https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html#torchvision.models.detection.fasterrcnn_resnet50_fpn`

    - 代码: torchvision.models.detection.fasterrcnn.py, 其中有权重下载链接

### [数据集](https://docs.pytorch.org/vision/stable/datasets.html):`https://docs.pytorch.org/vision/stable/datasets.html`

- 数据集加载器:[VOCDetection](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.VOCDetection.html):`https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.VOCDetection.html`
    
    - 代码: torchvision.datasets.voc.py, 其中有数据集下载链接
    - 类别: torchvision.models._meta.py, 其中有各数据集的类别名称

- 画框: [draw_bounding_boxes](https://docs.pytorch.org/vision/stable/utils.html):`https://docs.pytorch.org/vision/stable/utils.html`

### [github仓库](https://github.com/pytorch/vision):`https://github.com/pytorch/vision`