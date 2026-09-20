# Experiment log

A running record of each training run: what changed, what happened, and why.
Kept alongside the code because the *why* is the actual learning outcome of
this project, not just the final model.

## Round 1 -- smoke test

`configs/smoke_test.yaml`, a few hundred/thousand steps, single environment.
Purpose was purely to prove the pipeline (env, curriculum, PPO, Stockfish)
runs end to end. Not a real training attempt.

## Round 2 -- baseline, single environment

- Config: `curriculum.start_level=0` mapped directly to Stockfish Skill
  Level 0 (no bootstrap stage existed yet). `material_reward_scale=0`,
  `ent_coef` unset (SB3 default ~0). `n_envs=1`.
- 2,000,000 timesteps, ~9-16 fps, ~8h49m wall clock.
- **Result: 0 wins across 320 evaluation games** against real Stockfish
  Skill Level 0. `explained_variance` reached a healthy ~0.73 (the value
  function learned to predict outcomes reasonably well) but the policy
  never converted that into wins.
- Takeaway: a fixed, already-competent opponent from step 0 gives almost no
  positive signal to learn from -- classic sparse-reward cold start problem.

## Round 3 -- bootstrap stage, entropy bonus, repetition penalty

Added: level 0 = uniform-random-move opponent (bootstrap, needs no engine
at all); `material_reward_scale=0.01`; `ent_coef=0.01`; a
`repetition_penalty=0.05` per-move term; `n_envs=8` (parallel Stockfish
processes).

- 3,000,000 timesteps, ~9-60+ fps depending on stage, **only 53 minutes**
  wall clock (parallelism + the engine-free bootstrap stage are both much
  faster).
- **Result: never promoted past level 0**, not even once. Across 19,559
  games vs. the random mover: 80.1% draws, 14.0% wins, 5.8% losses -- a
  rolling win rate that peaked at 0.66 and never reached the 0.70
  promotion threshold. `explained_variance` collapsed to ~0 and
  `approx_kl`/`clip_fraction` spiked near the end (policy instability).
- Diagnosis: the model draws constantly even against pure randomness
  because it has no search/calculation ability -- it can get a material
  edge but often can't force checkmate. The entropy bonus and/or repetition
  penalty likely contributed to the late-run instability (both were new
  vs. round 2).

## Round 4 -- lower threshold, dial back entropy/repetition, resilience fixes

Kept the bootstrap stage; lowered `promotion_threshold` to 0.60; cut
`ent_coef` to 0.001; set `repetition_penalty=0.0` (removed); added
`claim_draw=True` everywhere a game's end is checked (`ChessEnv`,
`watch.py`, `browser_bot.py`) so a shuffling policy can't stall an episode
indefinitely; made `ChessEnv._play_opponent_move` retry/restart the
Stockfish engine on `TimeoutError`/`EngineError` instead of hanging the
whole `SubprocVecEnv`.

- Target 4,000,000 timesteps. Did promote once (level 0 -> level 1 =
  Stockfish Skill Level 0), quite early.
- Then plateaued hard against real Stockfish: across 58,456 games the score
  crept from 0.42% to 2.51% over six time-chunks -- a real but very slow
  upward trend, almost entirely from more draws, not more wins.
- Operational lesson learned the hard way: **the training machine going to
  sleep mid-run corrupts the `SubprocVecEnv` workers' pipes and hangs the
  whole process** (looks alive, uses ~0 CPU) -- not recoverable by the
  in-process engine-restart fix, since that only covers a single worker's
  Stockfish link, not the main process's IPC to its workers. Recovered via
  `--resume` from the last good checkpoint each time; don't let the machine
  sleep during a run. Also found and fixed a `reset_num_timesteps` bug:
  SB3's `.learn()` resets the step counter to 0 by default even after
  `.load()` restores it, which silently overwrote earlier checkpoint files
  at colliding step-count filenames on resume. Fixed by passing
  `reset_num_timesteps=not args.resume`.

A second `reset_num_timesteps=False` subtlety, found only once round 5b
overshot its target by ~3.3M steps: SB3 doesn't treat `total_timesteps` as
an absolute target when resuming -- it internally *adds* the checkpoint's
already-completed step count to whatever value is passed. `--timesteps
6000000` on a checkpoint already at ~4.06M silently became a target of
~10.06M. Fixed by pre-subtracting `model.num_timesteps` before calling
`.learn()`, verified with a fresh 512-step run resumed to an absolute
target of 768 (correctly trains exactly 256 more, not 512 more).

## Round 5 -- depth-ramp curriculum + supervised pretraining

Two more changes: (1) inserted a depth-ramp between the random bootstrap
and real Stockfish -- levels 1-3 are full-strength Stockfish limited to
1/2/4 ply of lookahead via `chess.engine.Limit(depth=N)`, rather than
jumping straight to a time-limited "Skill Level 0" search (round 4 found
that jump was a bigger difficulty cliff than expected); (2) supervised
pretraining: extracted 1,000,000 (position, human move) pairs from
lichess.org's January 2013 game database (games with both players >=1500
Elo), trained `ChessCNN` as a plain move classifier
(`src/prepare_pgn_dataset.py`, `src/pretrain.py`), and warm-started a fresh
PPO run's feature extractor from those weights (`train.py
--pretrained-features`).

- Target 4,000,000 timesteps, **1h35m wall clock** (mostly spent in the
  fast depth-ramp stage). Pretraining itself: 5 epochs, move-prediction
  accuracy 26.7% -> 33.8% on held-out human positions.
- Promoted once (bootstrap -> depth-ramp level 1, depth=1). Plateaued
  there at a ~49-50% rolling score -- close to, but never crossing, the
  0.60 threshold. `explained_variance` finished at 0.62 (healthier than
  round 4's near-zero).
- Evaluated the final checkpoint against **real, undiluted** Stockfish
  Skill Level 0 (not depth-limited): 0 wins, 2 draws, 18 losses in 20
  games. Also tried augmenting inference with alpha-beta search
  (`src/search.py`, depth 2): 0 wins, 1 draw, 9 losses in 10 games --
  no clear improvement at that shallow a depth (search was also tested at
  depth 3, which was too slow to finish even one game in a practical time
  and was abandoned).
- Live test: had the model play one real opponent on lichess.org via
  `browser_bot.py --manual` (opens a browser, a human starts the game in
  it, the bot detects its color and takes over) -- **result: a draw**, the
  best live result so far. Direct observation from watching it: the model
  repeatedly avoided winning captures it should have taken.
- Diagnosis: `material_reward_scale=0.01` is likely too weak a signal
  relative to the sparse +-1 terminal reward for "capturing material is
  good" to be reliably learned. Raised to `0.1` (10x) for the next
  continuation, resuming from this checkpoint rather than restarting.

## Round 5b -- stronger material reward (in progress)

Resumed round 5's final checkpoint with `material_reward_scale` raised
from 0.01 to 0.1; everything else unchanged. Purpose: directly test
whether a stronger material signal fixes the observed
avoids-free-captures behavior. Results pending.

## Recurring, general lessons

- **Curriculum promotion math is easy to get wrong by intuition.** A 0.70
  win-rate bar sounds modest, but with draws worth 0.5, if wins are rare
  the achievable ceiling can sit well below the bar no matter how long
  training runs (round 3's 0.66 ceiling; round 4's <3% after 58k games).
  Always check the actual win/draw/loss mix, not just "is win rate going
  up".
- **A pure policy/value network without search struggles specifically to
  *finish* winning positions**, even once it reliably reaches them. This
  shows up as excess draws (round 3, round 5) and as literally fleeing
  from favorable captures (round 5's live game). It's the concrete,
  observed reason engines like AlphaZero pair a network with MCTS rather
  than using the network's raw output directly.
- **Changing the reward function and re-evaluating live/interactively (not
  just via aggregate win-rate) surfaces failure modes aggregate metrics
  miss.** The piece-avoidance behavior wasn't visible in the win/draw/loss
  tally the same way it was obvious watching one real game.
