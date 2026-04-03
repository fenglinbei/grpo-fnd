本项目尝试使用GRPO完成FND任务的分类

配置库：configs/exp

若使用vllm加速推理，请先启动vllm：
VLLM_SERVER_DEV_MODE=1 \
CUDA_VISIBLE_DEVICES=2,3 \
vllm serve ./models/Qwen3-0.6B \
  --served-model-name live-policy \
  --tensor-parallel-size 2 \
  --weight-transfer-config '{"backend":"nccl"}'

运行方式（使用qwen_grpo_v1.yaml）：
python -m src.main --config configs/exp/qwen_grpo_v1.yaml

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
