#!/usr/bin/env python
"""Interactive terminal sampler for a trained ELF checkpoint.

This is an unconditional sampler for the laptop TinyStories model we trained.
Typed prompts are recorded in the session log, but they do not condition the
model unless you load a conditional checkpoint/config.
"""

import argparse
import contextlib
import copy
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime

import jax

try:
    jax.distributed.initialize()
except (RuntimeError, ValueError):
    pass

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import jax.numpy as jnp
import numpy as np
import optax
from flax import jax_utils
from transformers import AutoTokenizer

from configs.config import apply_config_overrides, load_config_from_yaml, load_sampling_configs
from modules.model import ELF_models
from modules.t5_encoder import get_encoder
from utils.checkpoint_utils import find_latest_checkpoint, load_checkpoint
from utils.data_utils import get_pad_token_id
from utils.generation_utils import (
    _make_pmap_pair,
    _shard_noise,
    _shard_timesteps,
    mask_after_eos,
)
from utils.train_utils import TrainState


DEFAULT_CONFIG = "../outputs/elf_t-tinystories-stage3-20e/config.yml"
DEFAULT_CHECKPOINT = "../outputs/elf_t-tinystories-stage3-20e/checkpoint_5000"
TOKEN_RE = re.compile(r"[A-Za-z0-9']+")

logging.basicConfig(
    format="%(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
    force=True,
)
logger = logging.getLogger(__name__)
for _name in ("absl", "orbax", "tensorstore", "flax.training.checkpoints"):
    logging.getLogger(_name).setLevel(logging.ERROR)


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive CLI sampler for ELF checkpoints.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to config YAML.")
    parser.add_argument("--checkpoint_path", default=DEFAULT_CHECKPOINT, help="Checkpoint dir/file or output dir.")
    parser.add_argument("--config_override", action="append", default=[], help="Config override field=value.")
    parser.add_argument("--samples", type=int, default=4, help="Samples per generation command.")
    parser.add_argument("--seed", type=int, default=42, help="Initial sampling seed.")
    parser.add_argument("--steps", type=int, default=16, help="Sampling steps.")
    parser.add_argument("--batch_size", type=int, default=8, help="Total batch size used during generation.")
    parser.add_argument("--self_cond_cfg", type=float, default=3.0, help="Self-conditioning CFG scale.")
    parser.add_argument("--cfg", type=float, default=1.0, help="Classifier-free guidance scale.")
    parser.add_argument("--quiet_jax", action="store_true", help="Reduce JAX/backend logging.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "gpu", "metal", "tpu"),
        help=(
            "Device used for model/state initialization: auto (default backend), "
            "cpu, gpu, metal (alias of gpu), or tpu."
        ),
    )
    parser.add_argument(
        "--log_path",
        default=None,
        help="JSONL session log path. Defaults to <config.output_dir>/interactive_logs/<timestamp>.jsonl.",
    )
    parser.add_argument(
        "--use_cpu",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Deprecated alias for --device cpu/auto (use --use_cpu or --no-use_cpu).",
    )
    return parser.parse_args()


def resolve_init_device(device_name: str):
    """Resolve init device. Metal maps to JAX GPU backend."""
    requested = (device_name or "auto").lower()
    backend = "gpu" if requested == "metal" else requested
    if backend == "auto":
        return "auto", None
    try:
        devices = jax.local_devices(backend=backend)
    except Exception as exc:
        raise ValueError(
            f"Requested init device backend '{requested}' is unavailable. "
            f"Default backend is '{jax.default_backend()}'. Error: {exc}"
        ) from exc
    if not devices:
        raise ValueError(
            f"Requested init device backend '{requested}' has no local devices. "
            f"Default backend is '{jax.default_backend()}'."
        )
    return requested, devices[0]


def tokenize(text):
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]


def text_stats(texts):
    tokens = []
    bigrams = []
    repeats = 0
    pairs = 0
    prefixes = set()
    for text in texts:
        toks = tokenize(text)
        tokens.extend(toks)
        if toks:
            prefixes.add(tuple(toks[:5]))
        if len(toks) > 1:
            local_pairs = list(zip(toks, toks[1:]))
            bigrams.extend(local_pairs)
            pairs += len(local_pairs)
            repeats += sum(1 for a, b in local_pairs if a == b)

    top = Counter(tokens).most_common(8)
    return {
        "samples": len(texts),
        "avg_chars": round(sum(len(t) for t in texts) / max(1, len(texts)), 2),
        "avg_words": round(sum(len(tokenize(t)) for t in texts) / max(1, len(texts)), 2),
        "distinct1": round(len(set(tokens)) / max(1, len(tokens)), 4),
        "distinct2": round(len(set(bigrams)) / max(1, len(bigrams)), 4),
        "adjacent_repeat_ratio": round(repeats / max(1, pairs), 4),
        "prefix_diversity": round(len(prefixes) / max(1, len(texts)), 4),
        "top_tokens": top,
    }


class InteractiveSampler:
    def __init__(self, args):
        if args.quiet_jax:
            logging.getLogger("jax").setLevel(logging.WARNING)
            logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)

        self.args = args
        self.seed = args.seed
        self.samples = args.samples
        self.steps = args.steps
        self.cfg = args.cfg
        self.self_cond_cfg = args.self_cond_cfg
        self.batch_size = args.batch_size
        self.debug = True

        self.config = load_config_from_yaml(args.config)
        if args.config_override:
            self.config = apply_config_overrides(self.config, args.config_override)
        if self.config.sampling_configs_path:
            self.config.sampling_configs = load_sampling_configs(self.config.sampling_configs_path)
        if not self.config.sampling_configs:
            raise ValueError("No sampling config found.")
        self.sampling_config = copy.copy(self.config.sampling_configs[0])
        self.sampling_config.num_sampling_steps = [self.steps]
        self.sampling_config.cfgs = [self.cfg]
        self.sampling_config.self_cond_cfg_scales = [self.self_cond_cfg]

        self.config.num_samples = self.samples
        self.config.online_eval = False
        self.config.use_wandb = False

        self.num_local_devices = jax.local_device_count()
        self.effective_batch_size = max(1, self.batch_size // self.num_local_devices) * self.num_local_devices
        self.per_device_batch = self.effective_batch_size // self.num_local_devices

        selected_device = "cpu" if args.use_cpu is True else args.device
        self.init_device_mode, init_device = resolve_init_device(selected_device)
        init_ctx = jax.default_device(init_device) if init_device is not None else contextlib.nullcontext()
        logger.info(
            "JAX default backend: %s | init device mode: %s",
            jax.default_backend(),
            self.init_device_mode,
        )

        logger.info("Loading tokenizer and model config...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name or self.config.encoder_model_name)
        self.pad_token_id = get_pad_token_id(self.tokenizer, self.config.pad_token)
        self.eos_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 1
        encoder_config, _, _ = get_encoder(self.config.encoder_model_name, jnp.float32)
        self.d_model = encoder_config.d_model

        logger.info("Initializing %s template...", self.config.model)
        rng = jax.random.PRNGKey(self.config.seed)
        rng, init_rng, dropout_rng = jax.random.split(rng, 3)
        input_dim = 2 * self.d_model if self.config.self_cond_prob > 0 else self.d_model
        with init_ctx:
            dummy_x = jnp.ones((1, self.config.max_length, input_dim))
            dummy_t = jnp.ones((1,))
            dummy_self_cond_cfg_scale = (
                jnp.ones((1,)) if self.config.num_self_cond_cfg_tokens > 0 else None
            )
            model = ELF_models[self.config.model](
                text_encoder_dim=self.d_model,
                max_length=self.config.max_length,
                attn_drop=self.config.attn_dropout,
                proj_drop=self.config.proj_dropout,
                num_time_tokens=self.config.num_time_tokens,
                num_self_cond_cfg_tokens=self.config.num_self_cond_cfg_tokens,
                vocab_size=self.tokenizer.vocab_size,
                num_model_mode_tokens=self.config.num_model_mode_tokens,
                bottleneck_dim=self.config.bottleneck_dim,
            )
            params = model.init(
                init_rng,
                x=dummy_x,
                t=dummy_t,
                deterministic=True,
                self_cond_cfg_scale=dummy_self_cond_cfg_scale,
            )
            state = TrainState.create(
                apply_fn=model.apply,
                params=params["params"],
                tx=optax.adamw(learning_rate=1e-4),
                dropout_rng=dropout_rng,
                ema_params1=copy.deepcopy(params["params"]),
            )

        self.param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
        checkpoint_path = find_latest_checkpoint(args.checkpoint_path) or args.checkpoint_path
        logger.info("Loading checkpoint: %s", checkpoint_path)
        state, _ = load_checkpoint(checkpoint_path, state)
        self.state = jax_utils.replicate(state)
        self.state_unreplicated = jax_utils.unreplicate(self.state)
        self.model_params_replicated = jax_utils.replicate(self.state_unreplicated.ema_params1)
        self.p_generate, self.p_decode_ids = _make_pmap_pair(
            self.state_unreplicated.apply_fn,
            self.config,
            self.sampling_config,
            self.cfg,
            self.self_cond_cfg,
        )

        self.log_path = self._build_log_path(args.log_path)
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        logger.info("Interactive log: %s", self.log_path)

    def _build_log_path(self, log_path):
        if log_path:
            return os.path.abspath(os.path.expanduser(log_path))
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return os.path.abspath(
            os.path.join(self.config.output_dir, "interactive_logs", f"session_{stamp}.jsonl")
        )

    def print_debug(self):
        print("\nDebug")
        print(f"  model: {self.config.model} ({self.param_count:,} params)")
        print(f"  checkpoint step/epoch: {int(self.state_unreplicated.step)}/{int(self.state_unreplicated.epoch)}")
        print(
            f"  backend/devices: {jax.default_backend()} / {self.num_local_devices} "
            f"(init mode: {self.init_device_mode})"
        )
        print(f"  max_length: {self.config.max_length}, d_model: {self.d_model}")
        print(f"  samples: {self.samples}, batch_size: {self.effective_batch_size}, steps: {self.steps}")
        print(f"  sampler: {self.sampling_config.sampling_method}, cfg: {self.cfg}, sc-cfg: {self.self_cond_cfg}")
        print(f"  log_path: {self.log_path}\n")

    def update_generation_settings(self):
        self.config.num_samples = self.samples
        self.sampling_config.num_sampling_steps = [self.steps]
        self.sampling_config.cfgs = [self.cfg]
        self.sampling_config.self_cond_cfg_scales = [self.self_cond_cfg]

    def rebuild_sampler(self):
        self.update_generation_settings()
        self.p_generate, self.p_decode_ids = _make_pmap_pair(
            self.state_unreplicated.apply_fn,
            self.config,
            self.sampling_config,
            self.cfg,
            self.self_cond_cfg,
        )

    def generate(self, user_input=""):
        self.update_generation_settings()
        t0 = time.time()
        all_generated = []
        generation_time = 0.0
        decode_time = 0.0
        processed = 0
        num_batches = (self.samples + self.effective_batch_size - 1) // self.effective_batch_size
        rng = jax.random.PRNGKey(self.seed)

        for batch_idx in range(num_batches):
            remaining = self.samples - processed
            if remaining <= 0:
                break
            current_total_batch = min(self.effective_batch_size, remaining)
            current_total_batch = (
                (current_total_batch + self.num_local_devices - 1) // self.num_local_devices
            ) * self.num_local_devices
            current_per_device = current_total_batch // self.num_local_devices

            batch_rng = jax.random.fold_in(rng, batch_idx)
            noise_rng, t_rng = jax.random.split(batch_rng)
            device_rngs = jax.random.split(noise_rng, self.num_local_devices)
            t_steps_sharded = _shard_timesteps(
                t_rng,
                self.num_local_devices,
                self.steps,
                self.sampling_config.time_schedule,
                self.config,
            )
            z_sharded = _shard_noise(
                device_rngs,
                self.num_local_devices,
                current_per_device,
                self.config.max_length,
                self.d_model,
                self.config.denoiser_noise_scale,
            )

            gen_start = time.time()
            latent_sharded = self.p_generate(
                model_params=self.model_params_replicated,
                rng=device_rngs,
                z=z_sharded,
                t_steps=t_steps_sharded,
                cond_seq=None,
                cond_seq_mask=None,
            )
            latent_sharded.block_until_ready()
            generation_time += time.time() - gen_start

            dec_start = time.time()
            predicted_ids_sharded = self.p_decode_ids(
                z=latent_sharded,
                model_params=self.model_params_replicated,
                t_final_val=t_steps_sharded[:, -1],
            )
            predicted_ids_sharded.block_until_ready()
            decode_time += time.time() - dec_start

            predicted_ids = predicted_ids_sharded.reshape(-1, predicted_ids_sharded.shape[-1])
            predicted_ids = mask_after_eos(
                predicted_ids,
                eos_token_id=self.eos_token_id,
                pad_token_id=self.pad_token_id,
            )
            for i in range(predicted_ids.shape[0]):
                if processed >= self.samples:
                    break
                text = self.tokenizer.decode(np.array(predicted_ids[i]), skip_special_tokens=True)
                all_generated.append(text)
                processed += 1

        stats = text_stats(all_generated)
        total_time = time.time() - t0
        event = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "user_input": user_input,
            "seed": self.seed,
            "samples": self.samples,
            "steps": self.steps,
            "cfg": self.cfg,
            "self_cond_cfg": self.self_cond_cfg,
            "generation_time_sec": round(generation_time, 4),
            "decode_time_sec": round(decode_time, 4),
            "total_time_sec": round(total_time, 4),
            "stats": stats,
            "generated": all_generated,
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

        print(f"\nSeed {self.seed} | samples={self.samples} | steps={self.steps}")
        for idx, text in enumerate(all_generated, 1):
            print(f"\n[{idx}] {text}")
        print("\nStats")
        print(f"  generation/decode/total: {generation_time:.2f}s / {decode_time:.2f}s / {total_time:.2f}s")
        print(f"  avg_words: {stats['avg_words']}, distinct1: {stats['distinct1']}, distinct2: {stats['distinct2']}")
        print(f"  adjacent_repeat_ratio: {stats['adjacent_repeat_ratio']}, prefix_diversity: {stats['prefix_diversity']}")
        print("  top_tokens:", ", ".join(f"{tok}:{cnt}" for tok, cnt in stats["top_tokens"]))

        self.seed += 1


def print_help():
    print(
        """
Commands
  /gen [text]       Generate samples. Text is logged but does not condition this unconditional model.
  <enter>           Generate samples.
  /samples N        Set samples per generation. /sample N also works.
  /steps N          Set sampling steps.
  /seed N           Set next seed.
  /cfg X            Set CFG scale.
  /sccfg X          Set self-conditioning CFG scale.
  /debug            Print model/checkpoint/runtime settings.
  /help             Show this help.
  /quit             Exit.
"""
    )


def main():
    args = parse_args()
    sampler = InteractiveSampler(args)
    print("\nELF interactive sampler ready.")
    print("This checkpoint is unconditional: prompts are logged, not used as conditioning.")
    print("Press Enter or type /gen to sample. Type /help for commands.\n")
    sampler.print_debug()

    while True:
        try:
            raw = input("elf> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break

        if raw in {"/quit", "/q", "quit", "exit"}:
            print("bye")
            break
        if raw in {"", "/gen"} or raw.startswith("/gen "):
            sampler.generate(raw[5:].strip() if raw.startswith("/gen ") else "")
            continue
        if raw == "/help":
            print_help()
            continue
        if raw == "/debug":
            sampler.print_debug()
            continue

        parts = raw.split(maxsplit=1)
        if len(parts) == 2 and parts[0] in {"/sample", "/samples", "/steps", "/seed", "/cfg", "/sccfg"}:
            try:
                if parts[0] in {"/sample", "/samples"}:
                    sampler.samples = max(1, int(parts[1]))
                elif parts[0] == "/steps":
                    sampler.steps = max(2, int(parts[1]))
                elif parts[0] == "/seed":
                    sampler.seed = int(parts[1])
                elif parts[0] == "/cfg":
                    sampler.cfg = float(parts[1])
                    sampler.rebuild_sampler()
                elif parts[0] == "/sccfg":
                    sampler.self_cond_cfg = float(parts[1])
                    sampler.rebuild_sampler()
                print("updated")
            except ValueError as exc:
                print(f"invalid value: {exc}")
            continue

        print("Unknown command. Type /help, /gen, or press Enter to sample.")


if __name__ == "__main__":
    main()
