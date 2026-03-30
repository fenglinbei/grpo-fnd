本项目尝试使用GRPO完成FND任务的分类

配置库：configs/exp

运行方式（使用qwen_grpo_v1.yaml）：
python -m src.main --config configs/exp/qwen_grpo_v1.yaml

模型下载（Qwen/Qwen3-0.6B）：
mkdir models
modelscope download --model Qwen/Qwen3-0.6B --local_dir ./models/Qwen3-0.6B
