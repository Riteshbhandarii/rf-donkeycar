# rf-donkeycar

Reinforcement learning for a self-driving DonkeyCar. Models are trained in the Donkey simulator (`gym_donkeycar`) with stable-baselines3 and exported to ONNX to run on the real car.

**Result:** a SAC agent trained for 150k steps on randomly generated roads drives the **Warren track, which it never saw in training, for 3 full laps** in a deterministic test.

## Results

All runs use SAC with a CNN policy and continue from the previous run (`--load`). Results are from `--test`, which drives deterministically without training noise or image jitter.

| Run | Steps | Change | Test result |
|---|---|---|---|
| `sac_a1_baseline` | 0 → 52k | Course pipeline, default settings | Takes the first two turns of a generated road |
| `sac_a2_steer` | 50k → 74k | Steering penalty 0.10 → 0.30 | Drives the road well until it hits something |
| `sac_a3_road3` | 70k → 100k | New generated roads | 2 full laps on an unseen generated road; most of a lap on Warren |
| `sac_a4_150k` | 100k → 150k | 3 more generated roads | **3 full laps on Warren** |

Warren is held out: it is used only for testing.

## How it works

```
Simulator 640×480 RGB
  → area-average to 160×120 (matches the real camera)
  → crop top 80 rows, grayscale            preprocessing.py, shared with the car
  → 40×160 image → SAC policy → [steering, throttle]
```

- **Observation:** 40×160 grayscale. The same `preprocessing.py` runs in training and in the car's ONNX inference, so both see identical images.
- **Reward:** staying near the centre line at a target speed, plus a small survival bonus, minus a penalty on *changes* in steering. Leaving the track ends the episode with −1.
- **Robustness:** during training only, the camera pose and image appearance (brightness, contrast, noise, glare) are randomised.
- **Generalisation:** the road is regenerated every ~15–20k steps so the agent learns to drive rather than memorise one road.

## Findings

- **Smooth steering.** At a steering penalty of 0.10, a full left-to-right flip cost about 0.18 reward against a best step of about 1.05, and the agent steered bang-bang. At 0.30 the flip costs about 0.54 while a steady turn stays free; the deterministic policy became smooth.
- **Road rotation generalises.** Episode length fell 54% after the first road switch (42k steps) but only 20% after a later one (85k).
- **Control rate on macOS.** Profiling showed about 53% of each step in PyTorch GPU work and 38% waiting for the next simulator frame, because both share the GPU. Low Power Mode cut training from about 13.5 to 7.5 steps/s.
- **Judge by `--test`.** Training always looks wobbly because SAC samples random actions; the deterministic policy is what the car runs.

## Repository layout

| Path | Contents |
|---|---|
| `project/mycar/train_sac.py` | SAC training and testing (real-car pipeline) |
| `project/mycar/preprocessing.py` | Crop and grayscale, shared by training and the car |
| `project/mycar/export_onnx.py` | Export a BC, PPO or SAC model to ONNX and verify it against PyTorch |
| `project/mycar/train_sac_zoo.py` | Earlier SAC experiments: RL Zoo preset, alternative rewards, fine-tune overrides |
| `project/mycar/train_ppo.py` | PPO training and testing |
| `train_bc.py` | Behavioural cloning in PyTorch |
| `project/mycar/compare_runs.py` | Plot reward, episode length and speed of several runs on one image |
| `project/mycar/manage.py`, `calibrate.py`, `config.py`, `myconfig.py` | DonkeyCar car application and configuration |
| `project/mycar/models/` | Trained models, ONNX exports and TensorBoard logs |
| `dockerfile`, `compose.yml`, `compose.mac.yml` | Docker environment (Linux/CUDA) |
| `MAC-SETUP.md` | Running natively on macOS with the Apple GPU |

## Getting started

Install the dependencies (`requirements.txt`), or use Docker on Linux. On macOS, follow `MAC-SETUP.md`. Start the simulator and leave it at its main menu; the scripts load the requested scene themselves.

```bash
cd project/mycar

# Test the 150k model on Warren
python train_sac.py --test --load models/sac_models/sac_a4_150k.zip \
    --env_name donkey-warren-track-v0 --test_steps 3000

# Train from scratch
python train_sac.py --output my_run

# Continue from a model (--timesteps counts extra steps)
python train_sac.py --load models/sac_models/sac_a4_150k.zip --timesteps 50000 --output my_run_2

# Export for the car
python export_onnx.py --algo sac --model models/sac_models/sac_a4_150k.zip --output my_model

# Training curves
tensorboard --logdir models/sac_models
```

Notes:

- Checkpoints (model and replay buffer) are saved every 10k steps. Stopping training with Ctrl+C does not save, so stop right after a checkpoint.
- The road can be changed during training with the simulator's exit button.
- When continuing with `--load`, the hyperparameters at the top of `train_sac.py` are ignored and the saved ones are used; reward and wrapper changes do apply.
- `--env_name` only takes effect when the simulator is at its main menu.

## Model files

| File | Description |
|---|---|
| `project/mycar/models/sac_models/sac_a4_150k.zip` | 150k SAC model (stable-baselines3) |
| `project/mycar/models/onnx_models/sac/sac_a4_150k.onnx` | Same model for the car. Input `[1, 1, 40, 160]` uint8, output `[1, 2]` steering and throttle |

Intermediate checkpoints and replay buffers are not committed (25 MB and 257 MB each).

## Limitations

- Not yet tested on the real car. Training ran at about 11 steps/s; DonkeyCar's default drive loop is 20 Hz (`config.py`), and the rate of the car's ONNX inference script is not confirmed.
- The agent can drift toward the road edge and not recover, and can take the wrong branch at a road fork.
- It drives cautiously, often with throttle below 0.1. Real top speed and steering range depend on the car's calibration.

## Credits

The real-car pipeline (`train_sac.py`, `preprocessing.py`, `export_onnx.py`, `myconfig.py`) was provided by the Turku University of Applied Sciences reinforcement learning course. This repository adds macOS support, the steering-penalty change, the training schedule and the experiments above.
