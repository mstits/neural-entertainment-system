# Project: Dreamer-Style World-Model RL on NES

## The pitch

Train an agent to beat Zelda **entirely inside a neural network's dream of the
game**. Use the real emulator only to collect seed experience and to validate
the final policy. The world model learns the NES's dynamics from ~24 hours of
real gameplay, then serves as a 1000×-faster surrogate environment for
policy training. A week-long training run on a single M-series Mac produces
an agent trained on the equivalent of **decades of subjective game time**.

If it works, it's the first published application of modern world-model RL
(Dreamer-v3 / DayDreamer class) to a complex 1986 NES ROM on consumer
hardware. The output artifact is a single 4-minute MP4 of the agent
speedrunning Zelda, captioned "trained inside a dream."

## Why this is interesting

* **Throughput math.** Real NES emulation via nes-py: ~100 agent-steps/sec
  per worker, ~1,700/sec aggregate across 16 workers. A compact
  transformer-based world model running on MPS/ANE can produce predictions
  at ~100,000+ imagined steps/sec on an M3 Max. That is a **50–100× hardware
  uplift**, and because imagined rollouts cost one forward pass per step
  (vs. cycle-accurate 6502 + PPU simulation), the gap grows further with
  batch size.
* **Data efficiency.** Dreamer-v3 solves Atari-100k (100k env steps) with a
  single recipe and hyperparameter set. The same recipe on Zelda should
  extract a good policy from **hours** of real play instead of weeks.
* **Research novelty.** Most world-model RL is demoed on DeepMind Control
  Suite / Atari-2600 / Minecraft. Nobody has publicly shown it on NES Zelda,
  a game with sprawling non-linear progression, 128 overworld screens, 9
  dungeons, an inventory system, and reward signals that only fire across
  minute-long horizons. Doing it well is a legitimate contribution.
* **Consumer hardware.** Everyone else runs RL on 8×A100 clusters. Doing
  this on an M-series laptop with Metal/MPS/ANE end-to-end is an aesthetic
  statement: *"the future of RL research is on the machine in front of you."*

## Core architecture

Three networks, all trained jointly (Dreamer-v3 blueprint, adapted):

### 1. World model (the dream)

```
obs_t ──► Encoder ─► z_t ─┐
                          ├─► Dynamics RNN ─► ẑ_{t+1}, r̂_{t+1}, d̂_{t+1}
          a_t ────────────┘                           │
                                                      └─► Decoder ─► ôbs_{t+1}
```

* **Encoder**: CNN (same stack we already use for policy) → categorical
  latent `z_t ∈ {32 classes}^32`. Discrete latents work far better than
  Gaussian for world models; DreamerV3 made this standard.
* **Dynamics (RSSM)**: recurrent GRU that predicts the next categorical
  latent given current state + action.
* **Decoder**: mirror-CNN → reconstructed observation + predicted reward +
  done flag. Reconstruction is what forces the latent to encode game state
  (hearts, enemies, Link's position, rupee count, etc.) without us hand-
  coding any of it.
* **RAM-as-auxiliary-loss** (our twist): also decode the 2 KB RAM
  snapshot from the latent. Our existing reward_functions use ~40 specific
  RAM bytes; predicting them explicitly grounds the latent in semantics
  the reward signal actually depends on, accelerating convergence.

### 2. Actor (policy)

Small MLP head: `z_t → action_logits`. Trained inside the world model via
imagined rollouts + PPO (same optimizer we already have).

### 3. Critic (value)

Another small head: `z_t → v(z_t)`. Uses symlog-transformed two-hot
regression as in DreamerV3 (much more stable than standard MSE over
wide reward ranges).

### Losses

| Component     | Loss                                      | Scale |
|---------------|-------------------------------------------|-------|
| Reconstruction| BCE on pixels + MSE on RAM bytes          | 1.0   |
| Reward pred   | Two-hot regression on symlog(r)           | 1.0   |
| Done pred     | BCE                                       | 1.0   |
| KL regularizer| Balanced free-bits to latent prior        | 0.1   |
| Actor         | PPO clip + entropy bonus                  | —     |
| Critic        | Symlog two-hot regression                 | —     |

## Training loop

```
repeat:
    # 1. Collect a short rollout in the REAL emulator (2–5 min of wall
    #    time). Writes (obs, action, reward, done) tuples to a replay
    #    buffer capped at 1M transitions (~8 GB at 84×84 uint8).
    real_rollout(n_steps=50_000)

    # 2. Train world model on the replay buffer for K gradient steps.
    #    Dense supervised learning. This is where Metal/MPS earns its
    #    keep. All three heads train simultaneously.
    train_world_model(steps=200_000, batch=64, seq_len=16)

    # 3. Imagine trajectories FROM replay starting states using the
    #    current world model + actor, for horizon H=15. Use those
    #    imagined trajectories to train actor + critic via PPO.
    #    This is the 1000× step: no emulation, pure GPU.
    imagined_updates(batch=128, horizon=15, updates=100_000)
```

After enough cycles, the actor is trained almost entirely in imagination,
grounded periodically by fresh real rollouts that keep the world model
from drifting.

## Milestones

### M0: Baseline reproducibility (1 week)
* Fork `dreamerv3-torch` (Jansen et al., reference PyTorch impl).
* Run the default Atari recipe on Breakout via gym-retro. Verify we hit
  published scores on our hardware. Proves the pipeline works.
* Output: scripts/world_model/baseline_atari.py

### M1: NES ROM adapter (1 week)
* Wire our existing `NESEnvironment` / `FrameTransport` as the rollout
  collector. Extend replay buffer to store 2 KB RAM alongside pixels.
* Train world model on 1 hour of random-policy Zelda rollouts. Visualize
  decoded reconstructions vs. real frames. By eye it should look like
  Zelda's overworld after ~50k gradient steps.
* Output: tensorboard logs showing reconstruction loss curves + side-by-
  side decoder-vs-real frame videos.

### M2: Policy training in imagination (2 weeks)
* Turn on actor/critic heads. Imagine trajectories, train PPO inside them.
* Validate: every 10k imagined updates, run the actor for 1k real steps
  and log cumulative reward. The key signal is **real-env reward climbing
  while 99% of training happens in imagination**.
* Output: plot of "real-env reward per hour of wall-clock training,"
  compared against our current real-only training baseline.

### M3: Ablations + paper-quality results (2 weeks)
* Ablate RAM-aux loss (how much does it help?).
* Ablate discrete vs. continuous latents.
* Ablate imagination horizon (5 vs. 15 vs. 50).
* Final run: 72-hour training, produce the "speedrunning inside a dream"
  demo video.
* Output: one PDF / blog post / video everyone shares.

**Total: 6 weeks, one person, one M-series Mac.**

## Risks & how we handle them

| Risk | Mitigation |
|---|---|
| World model diverges on long horizons | Use DreamerV3's two-hot symlog reward head + free-bits KL; these are the specific tricks that fixed divergence in the paper. If still unstable, shorten imagination horizon (proven fallback). |
| NES graphics are too structured: the model memorizes rather than learns dynamics | Strong data augmentation (random cropping, color jitter) on encoder input. Force generalization. |
| Zelda's long-horizon rewards (dungeon completion) are harder than Atari | Start with reward shaping we already have (exploration + item pickup + enemy kill), graduate to sparse reward once dense training converges. |
| World model reconstruction looks great but policy performs worse than our GA baseline | That's a real negative result and worth publishing too. Ablation findings alone (what helps / what doesn't on NES) are publishable. |
| Hardware can't hold batch size 64 × seq 16 × 3×84×84 in unified memory | Tested ceiling on M3 Max 128 GB is roughly batch=128 for this shape; we're well under. M1 Pro 32 GB needs batch=32. |

## Why now

* DreamerV3 (2024) proved one recipe generalizes across dozens of envs
  without per-env tuning. That removes the "RL is fragile art, not science"
  objection that kept this kind of work gated at big labs.
* MPS support in PyTorch 2.2+ is stable enough for long-running training.
  All the flaky MPS bugs of 2023 are gone.
* Apple silicon unified memory makes the batch=128 × seq=16 replay
  fetches trivially cheap: no PCIe bottleneck.
* The existing NES scaffolding (transport, profiles, reward funcs,
  checkpoint management, replay viewer, metrics) drops in as-is. We are
  not starting from zero; we are swapping one optimizer (genetic
  algorithm) for another (world-model-based PPO) underneath the same
  engine.

## Success criterion

One sentence: **"Trained inside a neural network's dream of Zelda, my AI
completes the first dungeon in under 5 minutes."**

One video, one blog post, one commit that says `world_model: v1 ships`.
Everything else is supporting evidence.

## File layout (when we build it)

```
src/world_model/
    encoder.py         # CNN → categorical latent
    dynamics.py        # RSSM (GRU)
    decoder.py         # pixel + RAM decoder
    actor.py           # policy head
    critic.py          # value head (symlog two-hot)
    replay.py          # on-disk replay buffer (pixels + RAM + actions)
    trainer.py         # world-model SGD loop
    imagine.py         # imagined-rollout generator + PPO on imagination
    schedule.py        # real-rollout / world-model / imagination cadence

scripts/world_model/
    baseline_atari.py  # M0: sanity-check on Atari
    train_zelda.py     # M1+: the main thing
    visualize.py       # side-by-side real vs decoded frames → MP4

tests/world_model/
    test_encoder_decoder_roundtrip.py
    test_dynamics_one_step_prediction.py
    test_imagination_matches_real_within_horizon.py
```

## Appendix: prior art we stand on

* **Dreamer v3** (Hafner et al. 2024). The recipe we follow almost verbatim.
* **DayDreamer** (Wu et al. 2022). World-model RL on real robots; shows
  the imagination-training paradigm scales beyond pixel games.
* **MuZero** (Schrittwieser et al. 2020). Not exactly a world model but
  the "learned dynamics + planning" framing is closely related.
* **PlaNet** (Hafner et al. 2019). The direct ancestor of Dreamer.
* **TransDreamer** (Chen et al. 2022). Transformer-based dynamics; may
  replace the RSSM GRU if convergence is slow.

All of these shipped with open-source reference code. We are not
inventing new RL theory; we are applying proven theory to a target nobody
has bothered to hit.
