import torch
from tqdm import tqdm

from src.sft.sft_builder import build_sft_batch


def train_sft_epoch(
    model,
    tokenizer,
    dataloader,
    optimizer,
    scheduler,
    device,
    global_step: int = 0,
    on_step_end=None,
):
    model.train()
    total_loss = 0.0
    total_steps = 0

    pbar = tqdm(dataloader, desc="SFT", dynamic_ncols=True)
    for batch_samples in pbar:
        input_ids, attention_mask, labels = build_sft_batch(
            tokenizer, batch_samples, device
        )

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        global_step += 1
        step_loss = float(loss.item())

        total_loss += step_loss
        total_steps += 1

        if on_step_end is not None:
            on_step_end(
                global_step=global_step,
                stage="sft",
                model=model,
                tokenizer=tokenizer,
                train_metrics={
                    "loss": step_loss,
                },
            )
            # callback 里可能会切到 eval，这里显式切回 train
            model.train()

        pbar.set_postfix(
            {
                "loss": f"{step_loss:.4f}",
                "gs": global_step,
            }
        )

    return {
        "loss": total_loss / max(1, total_steps),
        "global_step": global_step,
        "num_optimizer_updates": total_steps,
    }