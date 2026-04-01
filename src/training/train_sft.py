import torch
from tqdm import tqdm


def train_sft_epoch(
    model,
    tokenizer,
    dataloader,
    optimizer,
    scheduler,
    grad_accum_steps: int = 1,
    global_step: int = 0,
    on_step_end=None,
):
    model.train()
    total_loss = 0.0
    total_micro_steps = 0
    total_optimizer_updates = 0

    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(dataloader, desc="SFT", dynamic_ncols=True)

    for micro_step, batch_samples in enumerate(pbar, start=1):
        print(batch_samples)
        input_ids, attention_mask, labels = batch_samples

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        raw_loss = outputs.loss
        loss = raw_loss / grad_accum_steps

        loss.backward()

        total_loss += float(raw_loss.item())
        total_micro_steps += 1

        should_step = (
            micro_step % grad_accum_steps == 0
            or micro_step == len(dataloader)
        )

        if should_step:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad_norm = float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            total_optimizer_updates += 1

            current_lr = float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else 0.0

            if on_step_end is not None:
                on_step_end(
                    global_step=global_step,
                    stage="sft",
                    model=model,
                    tokenizer=tokenizer,
                    train_metrics={
                        "loss": float(raw_loss.item()),
                        "lr": current_lr,
                        "grad_norm": grad_norm,
                        "seq_len": int(input_ids.size(1)),
                        "grad_accum_steps": grad_accum_steps,
                        "micro_step": micro_step,
                        "optimizer_updates": total_optimizer_updates,
                    },
                )
                model.train()

        pbar.set_postfix(
            {
                "loss": f"{raw_loss.item():.4f}",
                "gs": global_step,
                "seq": int(input_ids.size(1)),
            }
        )

    return {
        "loss": total_loss / max(1, total_micro_steps),
        "global_step": global_step,
        "num_optimizer_updates": total_optimizer_updates,
    }