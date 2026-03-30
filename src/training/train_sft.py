import torch
from tqdm import tqdm

from src.sft.sft_builder import build_sft_batch

def train_sft_epoch(model, tokenizer, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    total_steps = 0

    pbar = tqdm(dataloader, desc="SFT")
    for batch_samples in pbar:
        input_ids, attention_mask, labels = build_sft_batch(tokenizer, batch_samples, device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        total_steps += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / max(1, total_steps)