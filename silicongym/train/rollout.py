"""Multi-turn rollouts and completion log-probs, shared by the custom GRPO (grpo_mt.py) and PPO (ppo.py) loops.

An episode is a list of turns. Each turn is one model call: the prompt is the chat-templated
environment observation (spec + tool history so far) and the completion is the model's action.
Turns are trained as separate (prompt, completion) sequences that share the episode's reward, so no
token-level stitching of conversations is needed. Single-turn mode is just one turn per episode.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from silicongym.data.task import Task
from silicongym.envs.verilog_env import VerilogEnv
from silicongym.train.model import format_prompt


@dataclass
class Turn:
    prompt_ids: list[int]
    comp_ids: list[int]
    text: str


@dataclass
class Episode:
    task: Task
    turns: list[Turn] = field(default_factory=list)
    env_reward: float = 0.0     # what the simulator-based environment returned
    reward: float = 0.0         # the signal used for learning (equals env_reward for the verifiable source)
    info: dict = field(default_factory=dict)
    done: bool = False


def generate_batch(model, tok, prompts: list[str], max_new_tokens: int, temperature: float,
                   top_p: float = 0.95) -> list[Turn]:
    import torch
    enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    model.eval()
    with torch.no_grad():
        out = model.generate(**enc, do_sample=True, temperature=temperature, top_p=top_p,
                             max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id, use_cache=True)
    plen = enc["input_ids"].shape[1]
    turns = []
    for i in range(len(prompts)):
        prompt_ids = enc["input_ids"][i][enc["attention_mask"][i].bool()].tolist()
        comp = out[i, plen:].tolist()
        if tok.eos_token_id in comp:                       # keep the EOS, drop the padding after it
            comp = comp[: comp.index(tok.eos_token_id) + 1]
        elif tok.pad_token_id in comp:
            comp = comp[: comp.index(tok.pad_token_id)]
        turns.append(Turn(prompt_ids, comp, tok.decode(comp, skip_special_tokens=True)))
    return turns


def run_episodes(model, tok, tasks: list[Task], group_size: int, env_factory, source,
                 max_new_tokens: int, temperature: float) -> list[Episode]:
    """`group_size` episodes per task, generated in lockstep so each turn is one batched generate()."""
    episodes = [Episode(t) for t in tasks for _ in range(group_size)]
    envs = [env_factory() for _ in episodes]
    obs = [env.reset(ep.task) for env, ep in zip(envs, episodes)]
    max_turns = envs[0].max_turns
    for _ in range(max_turns):
        live = [i for i, ep in enumerate(episodes) if not ep.done]
        if not live:
            break
        turns = generate_batch(model, tok, [format_prompt(tok, obs[i]) for i in live],
                               max_new_tokens, temperature)
        for i, turn in zip(live, turns):
            episodes[i].turns.append(turn)
            obs[i], reward, done, info = envs[i].step(turn.text, n_tokens=len(turn.comp_ids))
            if done:
                ep = episodes[i]
                ep.done, ep.env_reward, ep.info = True, reward, info
                ep.reward = source.score(ep.task, info, reward)
    return episodes


def group_advantages(rewards: list[float], group_size: int, eps: float = 1e-4) -> list[float]:
    """GRPO advantage: (r - group mean) / (group std + eps); all-equal groups get exactly 0."""
    out = []
    for g in range(0, len(rewards), group_size):
        grp = rewards[g:g + group_size]
        mean = sum(grp) / len(grp)
        std = (sum((r - mean) ** 2 for r in grp) / len(grp)) ** 0.5
        out += [0.0 if std == 0 else (r - mean) / (std + eps) for r in grp]
    return out


def completion_forward(model, turns: list[Turn], value_head=None):
    """Forward a micro-batch of turns -> (logps [B,T], mask [B,T], values [B,T] or None) over completion tokens."""
    import torch
    device = model.device
    seqs = [t.prompt_ids + t.comp_ids for t in turns]
    width = max(map(len, seqs))
    pad = 0
    ids = torch.full((len(seqs), width), pad, dtype=torch.long)
    attn = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):                         # right padding keeps positions simple
        ids[i, : len(s)] = torch.tensor(s)
        attn[i, : len(s)] = 1
    ids, attn = ids.to(device), attn.to(device)
    out = model(input_ids=ids, attention_mask=attn, output_hidden_states=value_head is not None)
    max_c = max(len(t.comp_ids) for t in turns)
    logps = torch.zeros((len(turns), max_c), device=device)
    mask = torch.zeros((len(turns), max_c), device=device)
    values = torch.zeros((len(turns), max_c), device=device) if value_head is not None else None
    log_probs = torch.log_softmax(out.logits.float(), dim=-1)
    for i, t in enumerate(turns):
        p, c = len(t.prompt_ids), len(t.comp_ids)
        tgt = ids[i, p:p + c]
        logps[i, :c] = log_probs[i, p - 1:p - 1 + c].gather(1, tgt.unsqueeze(1)).squeeze(1)
        mask[i, :c] = 1
        if value_head is not None:
            values[i, :c] = value_head(out.hidden_states[-1][i, p - 1:p - 1 + c].float()).squeeze(-1)
    return logps, mask, values


def sample_tasks(tasks: list[Task], n: int, rng: random.Random) -> list[Task]:
    return rng.sample(tasks, min(n, len(tasks)))


def prepare_policy(base: str, init_adapters: list[str], load_4bit: bool, lora: dict):
    """4-bit base (+ merged SiliconLM chain) with a fresh trainable LoRA -> (tok, peft_model)."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    from silicongym.train.model import LORA_TARGETS, load_causal_lm
    tok, model = load_causal_lm(base, init_adapters, load_4bit)
    if load_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
        target_modules=lora.get("targets", LORA_TARGETS), task_type="CAUSAL_LM"))
    model.config.use_cache = True
    return tok, model
