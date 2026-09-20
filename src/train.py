"""Train a curriculum-learning chess agent (PPO) against Stockfish.

Usage:
    python src/train.py --config configs/default.yaml
    python src/train.py --config configs/smoke_test.yaml --timesteps 20000
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
from typing import Any

import torch
import yaml
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from curriculum import CurriculumManager
from env import ChessEnv
from policy import ChessCNN


def mask_fn(env: ChessEnv) -> Any:
    return env.action_masks()


def make_env(
    stockfish_path: str,
    material_reward_scale: float,
    repetition_penalty: float,
    engine_think_time: float,
) -> ActionMasker:
    env = ChessEnv(
        stockfish_path=stockfish_path,
        skill_level=0,
        material_reward_scale=material_reward_scale,
        repetition_penalty=repetition_penalty,
        engine_think_time=engine_think_time,
    )
    return ActionMasker(env, mask_fn)


def make_vec_env(
    stockfish_path: str,
    material_reward_scale: float,
    repetition_penalty: float,
    engine_think_time: float,
    n_envs: int,
) -> VecEnv:
    """Build ``n_envs`` independent (Stockfish subprocess + board) environments.

    Each parallel environment runs its own Stockfish process, so wall-clock
    throughput scales roughly with ``n_envs`` up to the machine's core count.
    Uses ``SubprocVecEnv`` for n_envs > 1 (each in its own OS process, needed
    since Stockfish's UCI I/O and PPO's env stepping would otherwise
    serialize on a single core) and ``DummyVecEnv`` for n_envs == 1 to avoid
    subprocess overhead.
    """
    env_fn = partial(make_env, stockfish_path, material_reward_scale, repetition_penalty, engine_think_time)
    if n_envs == 1:
        return DummyVecEnv([env_fn])
    return SubprocVecEnv([env_fn for _ in range(n_envs)])


class CurriculumCallback(BaseCallback):
    """Feeds finished-episode outcomes into a CurriculumManager and pushes
    skill-level promotions back down to the Stockfish opponent(s)."""

    def __init__(self, curriculum: CurriculumManager, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.curriculum = curriculum

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        for done, info in zip(dones, infos, strict=True):
            if not done or "result" not in info:
                continue
            result = self._classify_result(info)
            promoted = self.curriculum.record_result(result)
            if self.verbose:
                print(
                    f"[curriculum] level={self.curriculum.level} "
                    f"win_rate={self.curriculum.win_rate():.2f} result={result}"
                )
            if promoted:
                self.training_env.env_method("set_skill_level", self.curriculum.level)
                if self.verbose:
                    print(f"[curriculum] promoted -> skill level {self.curriculum.level}")
            if self.curriculum.is_max_level() and self.verbose:
                print("[curriculum] reached max skill level")
        return True

    @staticmethod
    def _classify_result(info: dict[str, Any]) -> str:
        agent_color = info["agent_color"]
        result = info["result"]  # "1-0", "0-1", or "1/2-1/2"
        if result == "1/2-1/2":
            return "draw"
        white_won = result == "1-0"
        agent_won = white_won == bool(agent_color)
        return "win" if agent_won else "loss"


def load_config(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a curriculum-RL chess agent against Stockfish.")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Path to a YAML config file."
    )
    parser.add_argument(
        "--timesteps", type=int, default=None, help="Override total_timesteps from the config."
    )
    parser.add_argument(
        "--start-level", type=int, default=None, help="Override the starting Stockfish skill level."
    )
    parser.add_argument(
        "--n-envs", type=int, default=None, help="Override the number of parallel Stockfish environments."
    )
    parser.add_argument(
        "--stockfish-path", type=str, default=None, help="Override the Stockfish binary path."
    )
    parser.add_argument("--log-dir", type=str, default=None, help="Override the log/checkpoint directory.")
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to a checkpoint .zip to resume training from."
    )
    parser.add_argument(
        "--pretrained-features",
        type=str,
        default=None,
        help="Path to a feature-extractor state dict from pretrain.py (ignored with --resume).",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_config(args.config)

    stockfish_path = args.stockfish_path or config["stockfish_path"]
    log_dir = Path(args.log_dir or config.get("log_dir", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    curriculum_cfg = config["curriculum"]
    start_level = args.start_level
    if start_level is None and args.resume:
        # Resuming a checkpoint should also resume the curriculum's skill
        # level, not silently restart the opponent at level 0.
        resumed_state = Path(args.resume).parent.parent / "curriculum_state.json"
        start_level = CurriculumManager.load_level(resumed_state)
        if start_level is not None:
            print(f"[curriculum] resuming at skill level {start_level} (from {resumed_state})")
    if start_level is None:
        start_level = curriculum_cfg["start_level"]

    curriculum = CurriculumManager(
        start_level=start_level,
        max_level=curriculum_cfg["max_level"],
        window_size=curriculum_cfg["window_size"],
        promotion_threshold=curriculum_cfg["promotion_threshold"],
        log_path=log_dir / "curriculum_log.csv",
    )

    n_envs = args.n_envs or config["training"].get("n_envs", 1)
    vec_env = make_vec_env(
        stockfish_path,
        config["reward"]["material_reward_scale"],
        config["reward"].get("repetition_penalty", 0.0),
        config["engine"]["think_time"],
        n_envs,
    )
    vec_env.env_method("set_skill_level", curriculum.level)

    policy_kwargs = {
        "features_extractor_class": ChessCNN,
        "features_extractor_kwargs": {"features_dim": config["policy"]["features_dim"]},
    }

    if args.resume:
        model = MaskablePPO.load(args.resume, env=vec_env)
    else:
        training_cfg = config["training"]
        model = MaskablePPO(
            "CnnPolicy",
            vec_env,
            learning_rate=training_cfg["learning_rate"],
            n_steps=training_cfg["n_steps"],
            batch_size=training_cfg["batch_size"],
            n_epochs=training_cfg["n_epochs"],
            gamma=training_cfg["gamma"],
            ent_coef=training_cfg.get("ent_coef", 0.0),
            policy_kwargs=policy_kwargs,
            verbose=1,
        )
        if args.pretrained_features:
            state_dict = torch.load(args.pretrained_features, map_location=model.device)
            model.policy.features_extractor.load_state_dict(state_dict)
            print(f"Warm-started features_extractor from {args.pretrained_features}")

    # CheckpointCallback counts calls to _on_step(), which fires once per
    # vec-env step (i.e. once per n_envs timesteps), not once per timestep.
    checkpoint_freq_calls = max(1, config["training"]["checkpoint_freq"] // n_envs)
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq_calls,
        save_path=str(log_dir / "checkpoints"),
        name_prefix="ppo_chess",
    )
    curriculum_callback = CurriculumCallback(curriculum, verbose=1)

    # total_timesteps is always the ABSOLUTE target step count, on a fresh
    # run or a resumed one. Two SB3 quirks to work around for that:
    #  1. .learn() resets the step counter to 0 on every call by default,
    #     even after .load() restores it -- reset_num_timesteps=False stops
    #     that (without it, resuming reruns from step 0, silently
    #     overwriting the earlier run's checkpoint files at the same labels).
    #  2. With reset_num_timesteps=False, SB3 then treats total_timesteps as
    #     an ADDITIONAL step count, internally adding model.num_timesteps to
    #     whatever is passed -- so it must be pre-subtracted here, or a
    #     resumed run silently trains for (target + steps-already-done).
    total_timesteps = args.timesteps or config["training"]["total_timesteps"]
    if args.resume:
        total_timesteps = max(0, total_timesteps - model.num_timesteps)
    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList([checkpoint_callback, curriculum_callback]),
        progress_bar=True,
        reset_num_timesteps=not args.resume,
    )

    final_path = log_dir / "checkpoints" / "ppo_chess_final.zip"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(final_path))
    print(f"Training complete. Final model saved to {final_path}")

    vec_env.close()


if __name__ == "__main__":
    main()
