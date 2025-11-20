
# 使用Python 3.8镜像作为基础镜像 https://docker.aityp.com/image/docker.io/library/python:3.8
# docker pull swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/library/python:3.8
# docker tag  swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/library/python:3.8  docker.io/library/python:3.8
FROM python:3.8

# 设置工作目录
WORKDIR /project

# 文件到工作目录
COPY vehicle_yolo/requirements.txt ./
# 安装依赖 （合并RUN减少镜像层）
RUN pip install --no-cache-dir torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r requirements.txt

RUN pip uninstall -y opencv-python && \
    pip install --no-cache-dir opencv-python-headless

COPY vehicle_yolo/ ./

# 指定容器启动时运行的命令
CMD ["python", "main.py"]
