import torch
import torch.nn.functional as F
from torch.optim import AdamW


# =========================
# 0) 一些超参数
# =========================
group_size = 8          # 每个 prompt 采样多少个候选输出
clip_eps = 0.2          # PPO / GRPO clipping range
kl_beta = 0.02          # KL penalty 系数
lr = 1e-6
num_update_epochs = 4   # 对同一批 rollout 做几轮更新
max_new_tokens = 256

optimizer = AdamW(policy_model.parameters(), lr=lr)


# =========================
# 1) 奖励函数：推理分类例子
# =========================
def compute_reward(generated_text, gold_label):
    """
    一个简单的规则奖励：
    - 标签正确：+1.0
    - 格式正确：+0.2
    - 否则不给
    """
    reward = 0.0

    pred_label = extract_answer_label(generated_text)   # 从 <answer>...</answer> 里抽标签
    has_format = check_format(generated_text)           # 是否包含 <think> 和 <answer>

    if pred_label == gold_label:
        reward += 1.0
    if has_format:
        reward += 0.2

    return reward


# =========================
# 2) 组内 advantage
# =========================
def compute_group_advantages(rewards, eps=1e-8):
    """
    rewards: shape [B, G]
    对每个样本内部做 group-relative 标准化
    """
    mean = rewards.mean(dim=1, keepdim=True)           # [B, 1]
    std = rewards.std(dim=1, keepdim=True)             # [B, 1]
    advantages = (rewards - mean) / (std + eps)       # [B, G]
    return advantages


# =========================
# 3) 采样函数：old policy 生成一组候选答案
# =========================
@torch.no_grad()
def rollout_group(old_policy_model, tokenizer, prompts):
    """
    prompts: list[str], 长度 B
    返回：
      sequences:       [B, G, T]
      prompt_lens:     [B]
      old_logprobs:    [B, G, T_gen]
      ref_logprobs:    [B, G, T_gen]
      decoded_texts:   list[list[str]]
    """
    B = len(prompts)
    all_sequences = []
    all_old_logprobs = []
    all_ref_logprobs = []
    all_texts = []
    prompt_lens = []

    for prompt in prompts:
        # tokenize prompt
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        prompt_len = prompt_ids.shape[1]
        prompt_lens.append(prompt_len)

        group_sequences = []
        group_old_logprobs = []
        group_ref_logprobs = []
        group_texts = []

        for _ in range(group_size):
            # 1) 用 old policy 采样完整序列
            seq = sample_sequence(
                model=old_policy_model,
                input_ids=prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                top_p=1.0,
                do_sample=True,
            )   # [1, T_total]

            # 2) 计算 old policy 对生成 token 的 logprob
            old_lp = get_token_logprobs(
                model=old_policy_model,
                sequence_ids=seq,
                prompt_len=prompt_len
            )   # [T_gen]

            # 3) 计算 ref policy 对同一条序列的 logprob（KL 用）
            ref_lp = get_token_logprobs(
                model=ref_model,
                sequence_ids=seq,
                prompt_len=prompt_len
            )   # [T_gen]

            text = tokenizer.decode(seq[0], skip_special_tokens=True)

            group_sequences.append(seq[0])
            group_old_logprobs.append(old_lp)
            group_ref_logprobs.append(ref_lp)
            group_texts.append(text)

        # pad 到统一长度
        padded_seq, padded_old_lp, padded_ref_lp = pad_group_tensors(
            group_sequences, group_old_logprobs, group_ref_logprobs
        )

        all_sequences.append(padded_seq)        # [G, T]
        all_old_logprobs.append(padded_old_lp)  # [G, T_gen]
        all_ref_logprobs.append(padded_ref_lp)  # [G, T_gen]
        all_texts.append(group_texts)

    sequences = pad_batch_groups(all_sequences)            # [B, G, T]
    old_logprobs = pad_batch_groups(all_old_logprobs)      # [B, G, T_gen]
    ref_logprobs = pad_batch_groups(all_ref_logprobs)      # [B, G, T_gen]

    return sequences, torch.tensor(prompt_lens), old_logprobs, ref_logprobs, all_texts


# =========================
# 4) 当前 policy 在固定 rollout 上重新打分
# =========================
def get_current_logprobs(policy_model, sequences, prompt_lens):
    """
    sequences:   [B, G, T]
    返回 current_logprobs: [B, G, T_gen]
    只取生成部分 token 的 logprob
    """
    B, G, T = sequences.shape
    flat_seq = sequences.view(B * G, T)  # [B*G, T]

    logits = policy_model(flat_seq).logits[:, :-1, :]     # 预测下一个 token
    target = flat_seq[:, 1:]                              # 真正下一个 token

    log_probs = F.log_softmax(logits, dim=-1)
    token_logprobs = torch.gather(
        log_probs, dim=-1, index=target.unsqueeze(-1)
    ).squeeze(-1)                                         # [B*G, T-1]

    # 只保留生成段；prompt 部分不参与 policy loss
    current_logprobs = extract_generated_part(
        token_logprobs, sequences, prompt_lens
    )                                                     # [B, G, T_gen]

    return current_logprobs


# =========================
# 5) GRPO loss
# =========================
def grpo_loss(
    current_logprobs,   # [B, G, T_gen]
    old_logprobs,       # [B, G, T_gen]
    ref_logprobs,       # [B, G, T_gen]
    advantages,         # [B, G]
    gen_mask,           # [B, G, T_gen], 1 表示有效生成 token
):
    """
    一个接近工程实现的版本：
    - 序列级 advantage 共享给该序列所有生成 token
    - token 级 ratio / clip
    - token 级 KL penalty
    """
    # [B, G, T_gen]
    log_ratio = current_logprobs - old_logprobs
    ratio = torch.exp(log_ratio)

    # [B, G, 1] -> broadcast 到 token 维
    adv = advantages.unsqueeze(-1)

    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv

    policy_obj = torch.minimum(unclipped, clipped)   # [B, G, T_gen]

    # 常见实现里 KL 可直接用 token logprob 差近似/构造惩罚项
    token_kl = current_logprobs - ref_logprobs       # [B, G, T_gen]

    # 只在生成 token 上求平均
    valid_count = gen_mask.sum().clamp_min(1.0)

    objective = ((policy_obj - kl_beta * token_kl) * gen_mask).sum() / valid_count

    # 优化器通常做最小化，所以取负号
    loss = -objective
    return loss


# =========================
# 6) 单个训练 step
# =========================
def train_step(batch):
    """
    batch:
      {
        "prompt": [...],
        "gold_label": [...]
      }
    """
    prompts = batch["prompt"]
    gold_labels = batch["gold_label"]

    # ---- A. 用 old policy rollout ----
    sequences, prompt_lens, old_logprobs, ref_logprobs, texts = rollout_group(
        old_policy_model=old_policy_model,
        tokenizer=tokenizer,
        prompts=prompts
    )

    # ---- B. 算奖励 ----
    B = len(prompts)
    rewards = torch.zeros(B, group_size, device=device)

    for b in range(B):
        for g in range(group_size):
            rewards[b, g] = compute_reward(
                generated_text=texts[b][g],
                gold_label=gold_labels[b]
            )

    # ---- C. 组内 advantage ----
    advantages = compute_group_advantages(rewards)   # [B, G]

    # ---- D. 固定 rollout，多轮更新当前 policy ----
    for _ in range(num_update_epochs):
        current_logprobs = get_current_logprobs(
            policy_model=policy_model,
            sequences=sequences,
            prompt_lens=prompt_lens
        )

        gen_mask = build_generation_mask(
            sequences=sequences,
            prompt_lens=prompt_lens,
            current_logprobs=current_logprobs
        )   # [B, G, T_gen]

        loss = grpo_loss(
            current_logprobs=current_logprobs,
            old_logprobs=old_logprobs,
            ref_logprobs=ref_logprobs,
            advantages=advantages,
            gen_mask=gen_mask,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
        optimizer.step()

    # ---- E. 同步 old policy ----
    old_policy_model.load_state_dict(policy_model.state_dict())

    return {
        "loss": loss.item(),
        "reward_mean": rewards.mean().item(),
        "reward_std": rewards.std().item(),
    }