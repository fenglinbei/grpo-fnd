import os
import traceback
from typing import Iterable, Tuple

import torch
from transformers import AutoModelForCausalLM
from vllm import LLMEngine

# vLLM 版本之间 finalize 名字有变化；worker 文档里现在展示的是 finalize_layerwise_reload，
# 但部分文档页也出现 finalize_layerwise_processing。
try:
    from vllm.model_executor.model_loader.reload import (
        initialize_layerwise_reload,
        finalize_layerwise_reload as finalize_layerwise_fn,
    )
except ImportError:
    from vllm.model_executor.model_loader.reload import (
        initialize_layerwise_reload,
        finalize_layerwise_processing as finalize_layerwise_fn,
    )


MODEL_PATH = "./models/Qwen3-0.6B"
DTYPE = torch.bfloat16
DEVICE = "cuda"


def get_vllm_runtime_model_and_model_config(llm: vllm.LLM):
    """
    尽量兼容不同 vLLM 版本的内部对象路径。
    官方示例文档里可以直接从
    llm.llm_engine.model_executor.driver_worker.worker.model_runner.model
    访问到底层模型。:contentReference[oaicite:2]{index=2}
    """
    engine = llm.llm_engine

    model_candidates = [
        lambda: engine.model_executor.driver_worker.worker.model_runner.model,
        lambda: engine.model_executor.driver_worker.model_runner.model,
    ]
    config_candidates = [
        lambda: engine.model_config,
        lambda: engine.vllm_config.model_config,
        lambda: engine.model_executor.driver_worker.worker.model_runner.model_config,
        lambda: engine.model_executor.driver_worker.model_runner.model_config,
    ]

    runtime_model = None
    model_config = None

    for fn in model_candidates:
        try:
            runtime_model = fn()
            if runtime_model is not None:
                break
        except Exception:
            pass

    for fn in config_candidates:
        try:
            model_config = fn()
            if model_config is not None:
                break
        except Exception:
            pass

    if runtime_model is None:
        raise RuntimeError("Cannot locate vLLM runtime model from LLM internals.")
    if model_config is None:
        raise RuntimeError("Cannot locate vLLM model_config from LLM internals.")

    return runtime_model, model_config


def iter_hf_named_params(
    hf_model: torch.nn.Module,
    *,
    limit: int | None = None,
) -> Iterable[Tuple[str, torch.Tensor]]:
    """
    显式 remove_duplicate=False，避免 tied weights（如 lm_head / embed_tokens）
    被 named_parameters() 默认去重掉。
    """
    count = 0
    for name, param in hf_model.named_parameters(remove_duplicate=False):
        # 这里保持 checkpoint-format 语义：传参数名 + tensor
        # 放在 CPU 上即可，接近“从外部接收权重再 load_weights”的路径。
        yield name, param.detach().to(device="cpu", dtype=DTYPE).contiguous()
        count += 1
        if limit is not None and count >= limit:
            break


def print_name_diff(hf_model: torch.nn.Module, vllm_model: torch.nn.Module, topk: int = 20):
    hf_names = [n for n, _ in hf_model.named_parameters(remove_duplicate=False)]
    vllm_names = [n for n, _ in vllm_model.named_parameters(remove_duplicate=False)]

    print(f"[INFO] HF param count   : {len(hf_names)}")
    print(f"[INFO] vLLM param count : {len(vllm_names)}")

    only_hf = [n for n in hf_names if n not in set(vllm_names)]
    only_vllm = [n for n in vllm_names if n not in set(hf_names)]

    print(f"[INFO] Names only in HF   : {len(only_hf)}")
    print(f"[INFO] Names only in vLLM : {len(only_vllm)}")

    if only_hf:
        print("[INFO] Sample names only in HF:")
        for n in only_hf[:topk]:
            print("  ", n)

    if only_vllm:
        print("[INFO] Sample names only in vLLM:")
        for n in only_vllm[:topk]:
            print("  ", n)


@torch.no_grad()
def local_reload_repro(
    model_path: str,
    *,
    limit: int | None = None,
    run_generate_sanity: bool = False,
):
    print(f"[INFO] Loading vLLM runtime model from: {model_path}")
    # enforce_eager=True 是为了避免 CUDAGraph 把真实错误位置遮住；
    # 这是 vLLM 官方 troubleshooting 的建议。:contentReference[oaicite:3]{index=3}
    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        tensor_parallel_size=1,
        dtype="bfloat16",
        enforce_eager=True,
        trust_remote_code=False,
        gpu_memory_utilization=0.6,
        max_model_len=1024,
    )

    vllm_model, model_config = get_vllm_runtime_model_and_model_config(llm)

    print(f"[INFO] Loading HF model from: {model_path}")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
        device_map="cpu",
        trust_remote_code=False,
    )
    hf_model.eval()

    print_name_diff(hf_model, vllm_model, topk=20)

    # 取一个已知参数，前后做个校验
    probe_name = "model.layers.0.self_attn.q_proj.weight"
    try:
        probe_before = vllm_model.get_parameter(probe_name).detach().float().cpu().view(-1)[:8].clone()
        print(f"[INFO] Probe before reload: {probe_name} -> {probe_before.tolist()}")
    except Exception as e:
        print(f"[WARN] Cannot read probe param before reload: {e}")
        probe_before = None

    print("[INFO] Starting local layerwise reload...")

    try:
        # 这就是 worker.update_weights(checkpoint-format) 的核心路径。:contentReference[oaicite:4]{index=4}
        with torch.device(DEVICE):
            initialize_layerwise_reload(vllm_model)
            vllm_model.load_weights(iter_hf_named_params(hf_model, limit=limit))
            finalize_layerwise_fn(vllm_model, model_config)

        torch.cuda.synchronize()
        print("[OK] Local layerwise reload finished successfully.")

    except Exception as e:
        print("[ERROR] Local layerwise reload failed.")
        print(f"[ERROR] {type(e).__name__}: {e}")
        traceback.print_exc()
        raise

    try:
        probe_after = vllm_model.get_parameter(probe_name).detach().float().cpu().view(-1)[:8].clone()
        print(f"[INFO] Probe after reload : {probe_name} -> {probe_after.tolist()}")

        if probe_before is not None:
            same = torch.allclose(probe_before, probe_after)
            print(f"[INFO] Probe unchanged? {same}")
    except Exception as e:
        print(f"[WARN] Cannot read probe param after reload: {e}")

    if run_generate_sanity:
        from vllm import SamplingParams
        out = llm.generate(
            ["Hello"],
            sampling_params=SamplingParams(
                temperature=0.0,
                max_tokens=16,
            ),
        )
        print("[INFO] Sanity generation output:")
        print(out[0].outputs[0].text)


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    # 先跑一个“极小子集”更容易定位是哪一批权重触发问题。
    # 建议顺序：1 -> 2 -> 10 -> 100 -> None(全量)
    local_reload_repro(
        MODEL_PATH,
        limit=10,              # 先小子集；稳定后改成 None 跑全量
        run_generate_sanity=False,
    )