from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Iterable, Optional

import requests
from loguru import logger
from torch import nn

from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLTrainerSendWeightsArgs,
    NCCLWeightTransferEngine,
)
from vllm.utils.network_utils import get_ip, get_open_port

from src.evaluation.live_vllm_sync_config import LiveVLLMSyncEvalConfig


@dataclass
class SyncState:
    initialized: bool = False
    last_synced_step: int = -1
    inference_world_size: int = 0
    world_size: int = 0
    master_address: Optional[str] = None
    master_port: Optional[int] = None
    rank_offset: int = 1
    trainer_group: Optional[object] = None


class LiveVLLMWeightSyncController:
    """
    负责把 trainer 当前内存模型的权重同步到 vLLM server。
    当前实现：HTTP control plane + NCCL data plane
    """

    def __init__(self, cfg: LiveVLLMSyncEvalConfig):
        self.cfg = cfg
        self.state = SyncState()

        if cfg.sync_backend != "nccl":
            raise NotImplementedError(
                f"Only NCCL backend is implemented in this controller, got: {cfg.sync_backend}"
            )

    # -------------------------
    # HTTP helpers
    # -------------------------
    def _url(self, path: str) -> str:
        return f"{self.cfg.base_url.rstrip('/')}{path}"

    def _post_json(self, path: str, payload: dict, timeout: int):
        resp = requests.post(self._url(path), json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp

    def _post(self, path: str, timeout: int):
        resp = requests.post(self._url(path), timeout=timeout)
        resp.raise_for_status()
        return resp

    def _get_json(self, path: str, timeout: int):
        resp = requests.get(self._url(path), timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    # -------------------------
    # vLLM server control plane
    # -------------------------
    def get_world_size(self) -> int:
        data = self._get_json("/get_world_size", timeout=10)
        return int(data["world_size"])

    def init_remote_engine(self, master_address: str, master_port: int, rank_offset: int, world_size: int):
        payload = {
            "init_info": dict(
                master_address=master_address,
                master_port=master_port,
                rank_offset=rank_offset,
                world_size=world_size,
            )
        }
        self._post_json("/init_weight_transfer_engine", payload, timeout=self.cfg.request_timeout_init_s)

    def update_remote_engine(self, names: list[str], dtype_names: list[str], shapes: list[list[int]], packed: bool):
        payload = {
            "update_info": dict(
                names=names,
                dtype_names=dtype_names,
                shapes=shapes,
                packed=packed,
            )
        }
        self._post_json("/update_weights", payload, timeout=self.cfg.request_timeout_update_s)

    def pause_generation(self):
        self._post("/pause", timeout=60)

    def resume_generation(self):
        self._post("/resume", timeout=60)

    # -------------------------
    # NCCL trainer-side init
    # -------------------------
    def ensure_initialized(self):
        if self.state.initialized:
            return

        inference_world_size = self.get_world_size()
        world_size = inference_world_size + 1  # +1 for trainer rank 0
        master_address = get_ip()
        master_port = get_open_port()
        rank_offset = 1

        if self.cfg.verbose:
            logger.info(
                "Initializing live vLLM weight sync | backend=nccl | master={}:{} | inference_world_size={} | world_size={}",
                master_address,
                master_port,
                inference_world_size,
                world_size,
            )

        # 先异步通知 server 侧 init，它会等待 trainer 侧 NCCL group 就绪
        init_thread = threading.Thread(
            target=self.init_remote_engine,
            args=(master_address, master_port, rank_offset, world_size),
            daemon=True,
        )
        init_thread.start()

        trainer_group = NCCLWeightTransferEngine.trainer_init(
            dict(
                master_address=master_address,
                master_port=master_port,
                world_size=world_size,
            )
        )

        init_thread.join()

        self.state.inference_world_size = inference_world_size
        self.state.world_size = world_size
        self.state.master_address = master_address
        self.state.master_port = master_port
        self.state.rank_offset = rank_offset
        self.state.trainer_group = trainer_group
        self.state.initialized = True

    # -------------------------
    # metadata + sync
    # -------------------------
    @staticmethod
    def _collect_named_parameters(model: nn.Module):
        named_params = list(model.named_parameters())
        names = []
        dtype_names = []
        shapes = []

        # 查看前20个参数名
        all_names = [name for name, _ in model.named_parameters()]
        logger.debug("lm_head.weight" in all_names)
        logger.debug("Last 10 parameter names: {}", all_names[-10:])
        for name, p in named_params[:20]:
            logger.debug("Sample param | name={} | dtype={} | shape={}", name, p.dtype, p.shape)

        for name, p in named_params:
            names.append(name)
            dtype_names.append(str(p.dtype).split(".")[-1])
            shapes.append(list(p.shape))
        return named_params, names, dtype_names, shapes

    def should_sync(self, global_step: int, force: bool = False) -> bool:
        if force:
            return True
        if self.cfg.sync_policy == "never":
            return False
        if self.cfg.sync_policy == "always":
            return True
        if self.cfg.sync_policy == "if_step_changed":
            return global_step != self.state.last_synced_step
        raise ValueError(f"Unsupported sync_policy: {self.cfg.sync_policy}")

    def sync_from_model(self, model: nn.Module, global_step: int, force: bool = False):
        """
        同步当前内存模型到 vLLM server。
        注意：这个函数假设当前没有其他生成请求在飞行；即便如此仍显式 pause/resume。
        """
        if not self.should_sync(global_step=global_step, force=force):
            if self.cfg.verbose:
                logger.info(
                    "Skip live vLLM sync | global_step={} already synced",
                    global_step,
                )
            return

        self.ensure_initialized()
        named_params, names, dtype_names, shapes = self._collect_named_parameters(model)

        if self.cfg.verbose:
            logger.info(
                "Syncing weights to live vLLM | step={} | param_count={} | packed={}",
                global_step,
                len(named_params),
                self.cfg.packed,
            )

        self.pause_generation()

        # 先起一个线程调用 server /update_weights，这个调用会阻塞等待 NCCL 广播真正发生
        update_thread = threading.Thread(
            target=self.update_remote_engine,
            args=(names, dtype_names, shapes, self.cfg.packed),
            daemon=True,
        )
        update_thread.start()

        trainer_args = NCCLTrainerSendWeightsArgs(
            group=self.state.trainer_group,
            packed=self.cfg.packed,
        )
        NCCLWeightTransferEngine.trainer_send_weights(
            iterator=iter(named_params),
            trainer_args=trainer_args,
        )

        update_thread.join()
        self.resume_generation()

        self.state.last_synced_step = global_step

        if self.cfg.verbose:
            logger.info("Live vLLM sync finished | step={}", global_step)