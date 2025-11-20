import docker
import random
import sseclient
import pytest
import tempfile
import os
import json
from typing import Dict, Any, Optional

# ============================
# 全局配置变量
# ============================
def random_subset_simple(input_list: list[Any]) -> list[Any]:
    return [item for item in input_list if random.random() <= 0.5]

def random_power_sub(input_list: list[Any]) -> list[Any]:
    return [
        random_subset_simple(input_list)
        for _ in range(random.randint(1, len(input_list)))
    ]

# Docker镜像名称
IMAGE_NAME = "wind-service-attack-text:latest"

# 文件映射测试用例
ENVIRONMENT_VARS = {
    "ATTACK_METHOD": "genetic",
    "POPULATION_SIZE": "100",
    "GENERATIONS": "20",
    "ATTACK_METHOD": "mixed",
    "CHAR_LEVEL_INTENSITY": "0.2",
    "SYNONYM_RATIO": "0.4",
}

# 环境变量测试用例
VOLUME_MOUNT = {
    # first one
    "/mnt/HDD1/wuhao/project/wind_service/resources/": {
        "bind": "/project/wind_service/resources/",
        "mode": "rw",
    },
    # seoncd one...
}


# 文件映射测试用例
VOLUME_MOUNT_TEST_CASES = random_power_sub(list(VOLUME_MOUNT.items()))
# 环境变量测试用例
ENVIRONMENT_VARS_TEST_CASES = random_power_sub(list(ENVIRONMENT_VARS.items()))

# GPU使用测试用例
GPU_TEST_CASES = [True]

# ============================
# 工具类
# ============================

class AttackTextValidator:
    def __init__(self, image_name: str = IMAGE_NAME):
        self.client = docker.from_env()
        self.image_name = image_name
        self.container = None

    def run_container(self, 
                     environment_vars: Dict[str, str],
                     volume_mounts: Dict[str, Dict[str, str]],
                     use_gpu: bool = False) -> sseclient.SSEClient:
        """运行容器并返回SSE客户端"""

        # 准备容器配置
        container_config = {
            "image": self.image_name,
            "environment": environment_vars,
            "volumes": volume_mounts,
            "detach": True,
        }

        # 如果启用GPU
        if use_gpu:
            container_config["device_requests"] = [
                docker.types.DeviceRequest(
                    count=-1,
                    capabilities=[['gpu']]
                )
            ]

        # 运行容器
        self.container = self.client.containers.run(**container_config)

        # 获取容器输出流
        stream = self.container.logs(stream=True, follow=True)

        # 创建SSE客户端
        return sseclient.SSEClient(stream)

    def cleanup(self):
        """清理容器"""
        if self.container:
            try:
                self.container.attach()
                self.container.remove()
            except:
                pass  # 忽略清理错误


def create_test_file(content: str) -> str:
    """创建临时测试文件"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        return f.name


def validate_sse_output(sse_client: sseclient.SSEClient):
    """验证SSE输出格式"""
    messages_received = 0

    try:
        for event in sse_client.events():
            if event.event == 'message' and event.data:
                messages_received += 1

                # 验证数据格式
                validate_sse_data_format(event.data)

                # 如果收到至少一条消息，认为测试通过
                if messages_received >= 1:
                    break

            elif event.event == 'error':
                # 错误信息也应该是SSE格式
                validate_sse_data_format(event.data)
                break
        else:
            pytest.fail(f"no messages recieved")

    except Exception as e:
        # 如果至少收到一条有效消息，则认为测试通过
        pytest.fail(f"No valid SSE messages received. Error: {e}")


def validate_sse_data_format(json_str: str):
    """
    验证JSON格式
    """
    json_data = json.loads(json_str)

    # 验证返回的数据结构是字典
    assert isinstance(json_data, dict), f"SSE data should be a JSON object, got: {type(json_data)}"



# ============================
# 测试类
# ============================

class TestEnvironmentVariables:
    """测试环境变量存在情况"""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        self.validator = AttackTextValidator()
        self.test_file_path = create_test_file("Test content for environment variable testing.")
        self.host_dir = os.path.dirname(self.test_file_path)
        yield
        self.validator.cleanup()

    @pytest.mark.parametrize("environment_vars", ENVIRONMENT_VARS_TEST_CASES)
    def test_environment_variables(self, environment_vars):
        """测试各种环境变量组合"""

        print(environment_vars)
        # 添加文件路径到环境变量
        environment_vars = dict(environment_vars)
        environment_vars["INPUT_FILE"] = os.path.join('/app/input', os.path.basename(self.test_file_path))

        # 运行容器并获取SSE客户端
        sse_client = self.validator.run_container(
            environment_vars=environment_vars,
            volume_mounts=VOLUME_MOUNT,
            use_gpu=True
        )

        # 验证SSE输出格式
        validate_sse_output(sse_client)


class TestVolumeMounts:
    """测试文件映射存在情况"""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        self.validator = AttackTextValidator()
        self.created_files = []
        yield
        self.validator.cleanup()

    @pytest.mark.parametrize("mount_case", VOLUME_MOUNT_TEST_CASES)
    def test_volume_mounts(self, mount_case):
        """测试各种文件映射组合"""
        # 运行容器并获取SSE客户端
        sse_client = self.validator.run_container(
            environment_vars={},
            volume_mounts=dict(mount_case),
            use_gpu=True
        )

        # 验证SSE输出格式
        validate_sse_output(sse_client)

    def test_with_nonexistent_file_mount(self):
        """测试不存在的文件映射"""
        environment_vars = {"ATTACK_METHOD": "char_level"}
        # 使用一个不存在的路径作为挂载
        volume_mounts = {
            "/nonexistent/path": {'bind': '/app/input', 'mode': 'rw'}
        }
        environment_vars["INPUT_FILE"] = "/app/input/nonexistent.txt"

        sse_client = self.validator.run_container(
            environment_vars=environment_vars,
            volume_mounts=volume_mounts,
            use_gpu=True
        )

        # 即使文件不存在，也应该有SSE格式的输出
        validate_sse_output(sse_client)

    def test_without_any_volume_mounts(self):
        """测试没有任何文件映射"""
        environment_vars = {"ATTACK_METHOD": "char_level"}
        # 不提供任何卷挂载
        volume_mounts = {}
        environment_vars["INPUT_FILE"] = "/app/input/nonexistent.txt"

        sse_client = self.validator.run_container(
            environment_vars=environment_vars,
            volume_mounts=volume_mounts,
            use_gpu=True
        )

        # 即使没有文件映射，也应该有SSE格式的输出
        validate_sse_output(sse_client)


class TestGPUUsage:
    """测试GPU使用情况"""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        self.validator = AttackTextValidator()
        self.test_file_path = create_test_file("Test content for GPU testing.")
        self.host_dir = os.path.dirname(self.test_file_path)
        yield
        self.validator.cleanup()

    @pytest.mark.parametrize("use_gpu", GPU_TEST_CASES)
    def test_gpu_usage(self, use_gpu):
        """测试GPU启用/禁用"""
        environment_vars = {"ATTACK_METHOD": "char_level"}
        volume_mounts = VOLUME_MOUNT
        environment_vars["INPUT_FILE"] = os.path.join('/app/input', os.path.basename(self.test_file_path))

        sse_client = self.validator.run_container(
            environment_vars=environment_vars,
            volume_mounts=volume_mounts,
            use_gpu=use_gpu
        )

        validate_sse_output(sse_client)


class TestCombinedCases:
    """测试组合情况"""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        self.validator = AttackTextValidator()
        self.test_file_path = create_test_file("Test content for combined testing.")
        self.host_dir = os.path.dirname(self.test_file_path)
        yield
        self.validator.cleanup()

    @pytest.mark.parametrize("environment_vars", ENVIRONMENT_VARS_TEST_CASES[:3])  # 只测试前3种环境变量
    @pytest.mark.parametrize("use_gpu", GPU_TEST_CASES)
    def test_combined_cases(self, environment_vars, use_gpu):
        """测试环境变量和GPU的组合"""
        environment_vars = dict(environment_vars)
        environment_vars["INPUT_FILE"] = os.path.join('/app/input', os.path.basename(self.test_file_path))

        sse_client = self.validator.run_container(
            environment_vars=environment_vars,
            volume_mounts=VOLUME_MOUNT,
            use_gpu=use_gpu
        )

        validate_sse_output(sse_client)


if __name__ == "__main__":
    # 可以直接运行测试
    pytest.main([__file__, "-v"])
