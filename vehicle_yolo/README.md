# 车辆识别场景 — API 使用示例与输入/输出样本

## 车辆识别攻击 `-attack-text:latest`

## 环境变量：
* `process`（必填）: 指定进程名称，目前车辆识别`adv`, `attack`, `defend`, `train`。
* `model`（必填）: 指定模型名称，支持枚举 `model_list` 中的值。
* `data`（必填）: 指定数据集，支持枚举 `data_list` 中的值。
* `input_path`（必填）: 指定输入路径，在此路径下有权重文件和数据集文件。
* `output_path`（必填）: 指定输出路径，在此路径下保存生成的对抗样本和防御训练的权重。
* `attack_method`（选填）: 指定攻击方法，若`process`为`adv`或`attack`则必填，支持枚举 `attack_method_list` 中的值。
* `defend_method`（选填）: 指定防御方法，若`process`为`defend`则必填，支持枚举 `defend_method_list` 中的值。
* `epochs`（选填，默认为100）：训练迭代次数，若`process`为`defend`或`train`有效。
* `batch`（选填，默认为16）：训练批处理大小，若`process`为`defend`或`train`有效。
* `device`（选填，默认为0）：使用哪个gpu。
* `selected_samples`（必填，默认数据集全部样本数据）: 本次使用的样本数。
* `epsilon`（选填，默认为各方法的默认值）：扰动强度参数，控制对抗扰动大小。
* `step_size`（选填，默认为各方法的默认值）：步长，迭代攻击的更新幅度
* `max_iterations`（选填，默认为各方法的默认值）：最大迭代次数
* `random_start`（选填，默认为各方法的默认值）：是否随机初始化扰动
* `loss_function`（选填，默认为各方法的默认值）：损失函数类型
* `optimization_method`（选填，默认为各方法的默认值）：优化方法

---

## 使用示例（命令行）

```bash
1）对抗样本生成：python main.py --task adv_sample_gen --model yolov5 --data imagenet --attack_method cw --weight_path xxx.pt --dataset_path xxx.zip --working_path xxx

2）攻击：python main.py --task attack --model yolov5 --data imagenet --attack_method cw --weight_path xxx.pt --dataset_path xxx.zip --working_path xxx

```

## 联系与版本

* 文档版本：V1.1
* 维护者：`wind_service` 安全团队

---

*文件生成于自动化文档工具，由开发者维护，若需同步到项目 `docs/` 目录请告知。*
