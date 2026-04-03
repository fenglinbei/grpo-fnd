本项目尝试使用GRPO完成FND任务的分类

配置库：configs/exp

若使用vllm加速推理，请先启动vllm：(启动前请确定端口是否空闲)
VLLM_SERVER_DEV_MODE=1 \
NCCL_CUMEM_HOST_ENABLE=0 \
NCCL_CUMEM_ENABLE=0 \
CUDA_VISIBLE_DEVICES=3 \
vllm serve ./models/Qwen3-0.6B \
  --served-model-name live-policy \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port 10909 \
  --weight-transfer-config '{"backend":"nccl"}'

运行方式（使用qwen3-0.6b_liar_vllm.yaml）：
NCCL_CUMEM_HOST_ENABLE=0 \
NCCL_CUMEM_ENABLE=0 \
python -m src.main --config configs/exp/qwen3-0.6b_liar_vllm.yaml

模型下载（Qwen/Qwen3-0.6B）：
mkdir models
modelscope download --model Qwen/Qwen3-0.6B --local_dir ./models/Qwen3-0.6B

环境安装（Linux下）：
conda create -p ../conda/grpo python=3.12
或者
conda create -n grpo python=3.12

cuda13.0:
pip3 install torch torchvision

cuda12.8:
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128

vllm:
pip install uv
uv pip install vllm --torch-backend=auto

其他依赖：
pip install -r requirements.txt
