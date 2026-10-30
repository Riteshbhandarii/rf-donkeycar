# rf-donkeycar

Self-driving experiments for DonkeyCar in the Donkey simulator (`gym_donkeycar`).

## What's here

| Path | What it is |
|---|---|
| `project/mycar/train_ppo.py` | PPO training and testing with stable-baselines3 |
| `project/mycar/train_sac.py` | SAC training and testing for the real-car pipeline (see [Real-car SAC](#real-car-sac-2026-10-30)) |
| `project/mycar/train_sac_zoo.py` | Earlier SAC script (Sessions 4–6): `--fast`, `--zoo`, `--reward`, `--lr/--gamma/--tau` overrides |
| `project/mycar/export_onnx.py` | Exports a trained BC/PPO/SAC model to ONNX for the car and checks PyTorch vs ONNX outputs |
| `project/mycar/compare_runs.py` | Plots reward, episode length and speed of PPO/SAC runs on one PNG |
| `train_bc.py` | Behavioural cloning training in PyTorch |
| `project/mycar/manage.py`, `train.py`, `calibrate.py`, `config.py`, `myconfig.py` | Standard DonkeyCar car app scripts and config |
| `project/mycar/preprocessing.py` | Crop and grayscale, shared by training and the car's ONNX inference |
| `project/mycar/models/ppo_models/ppo_session3/checkpoints/` | PPO checkpoints at 20k and 40k steps |
| `project/mycar/models/ppo_models/*/tb_logs/` | TensorBoard logs |
| `project/mycar/models/sac_models/<name>/` | SAC output: `checkpoints/` (`sac_donkey_<N>_steps.zip` + `_buffer.pkl`), `tb_logs/`, final model |
| `project/mycar/*.log` | Training and test run output |
| `project/mycar/eta.sh` | Prints training progress and ETA from a training log |
| `dockerfile`, `compose.yml`, `compose.mac.yml`, `entrypoint.sh` | Docker environment |
| `MAC-SETUP.md` | Running natively on macOS |
| `requirements.txt` | Python dependencies |

## Running the PPO model

Start the Donkey simulator first. The script connects to it on port 9091.

```
cd project/mycar
python -u train_ppo.py --test --load models/ppo_models/ppo_session3/checkpoints/ppo_donkey_40000_steps.zip
```

Add `--env_name donkey-warren-track-v0` for the Warren track. The default is `donkey-generated-roads-v0`.

Train a new model:

```
python train_ppo.py --output <model_name>
```

It saves checkpoints under `models/ppo_models/<model_name>/`.

View training curves:

```
tensorboard --logdir models/ppo_models/
```

## Session 4: SAC

Checklist:

- [ ] `train_sac.py` is in `project/mycar/` (newest version from ITSLearning)
- [ ] Simulator open, timescale 2-4, "Generated Road" scene
- [ ] Training started with `--output <name>`; it keeps running while you regen the track or switch scenes, as long as the simulator stays open
- [ ] TensorBoard open on `models/sac_models/`; then on `models/` to compare PPO and SAC
- [ ] New road generated after the car clears the track twice in a row (or if the road is bad, e.g. overlapping)
- [ ] Resumed once from a checkpoint (`--load`, buffer is picked up automatically)
- [ ] Tested with `--test` (no buffer needed)
- [ ] Wiggling on the centre line, or car stalls in `--test` but drove fine in training: change the reward, one modification at a time, retrain

Train (start the simulator first):

```
cd project/mycar
python -u train_sac.py --output sactest
```

Resume from a checkpoint (loads the matching `_buffer.pkl` from the same folder; use a new `--output` or the old name to overwrite):

```
python -u train_sac.py --load models/sac_models/sactest/checkpoints/sac_donkey_50000_steps.zip --output sactest2
```

Test (buffer not needed, so the model file can be copied alone):

```
python -u train_sac.py --test --load models/sac_models/sactest/checkpoints/sac_donkey_50000_steps.zip
```

TensorBoard, http://localhost:6006/ (add `--bind_all` inside Docker):

```
tensorboard --logdir models/sac_models/     # SAC only
tensorboard --logdir models/                # SAC vs PPO
```

Or without a browser, one PNG with reward, episode length and speed per run:

```
python compare_runs.py                                  # all runs under models/ppo_models and models/sac_models
python compare_runs.py models/ppo_models/ppo_session3 models/sac_models/sactest --out cmp.png
```

Both scripts log `rollout/avg_speed`, so the speed curves are comparable. Sharp drops after a track regen or a resume are normal; a drop that never recovers means a corrupted run, so resume from the last good checkpoint.

Reward (`train_sac_zoo.py` only): `--reward smooth` targets the "drives in training, stands still in --test" problem: stalling on track costs -0.1 instead of 0, and the +0.2 survival bonus only pays when moving near the centre line. Steering smoothness is in both rewards. It is opt-in; the default reward is the instructor's, unchanged.

## Real-car SAC (2026-10-30)

`train_sac.py`, `preprocessing.py`, `myconfig.py` and `export_onnx.py` are the course's real-car pipeline. The sim renders 640×480, the frame is area-averaged to 160×120 like the real camera, then `preprocessing.py` crops the top 80 rows and converts to grayscale, giving a 40×160 image. The same `preprocessing.py` runs on the car (ONNX runtime on the Jetson), so it must not change on one side only. Camera-pose and appearance jitter run in training only.

Trained natively on an M-series MacBook (MPS) with the `CourseSim2Mac` simulator.

### What changed from the handed-out script

| Commit | Change |
|---|---|
| `6406108` | macOS support only: MPS device, sim on `127.0.0.1`. Hyperparameters unchanged. |
| `089bdfe` | `steering_penalty` 0.10 → 0.30, `CHECKPOINT_FREQ` 25k → 10k |

The steering penalty is charged on the change in steering between steps. At 0.10 a full left-to-right flip cost about 0.18 reward against a best step of about 1.05, so the agent steered bang-bang. At 0.30 the flip costs about 0.54 while holding a steady turn stays free.

### Runs

Each run continues from the previous one with `--load`.

| Run | Steps | What changed | Deterministic `--test` result |
|---|---|---|---|
| `sac_a1_baseline` | 0 → 52k | Handed-out settings; new generated road at ~42k | 50k: takes the first two turns |
| `sac_a2_steer` | 50k → 74k | Steering penalty 0.30 | 70k: drives the road well until it hits something |
| `sac_a3_road3` | 70k → 100k | New road at 70k and ~87k | 100k: 2 full laps on a fresh generated road; on Warren most of a lap, then goes straight at one turn |
| `sac_a4_150k` | 100k → 150k | 3 more generated roads (switch every ~17k) | **150k: 3 full laps on Warren** |

Warren was never used for training, so it is the held-out track.

### Findings

- **Low Power Mode halves the control rate.** With it on, training ran at about 7.5 steps/s; off, about 13.5 steps/s. A 10 s profile of the training process showed about 53% of each step in PyTorch MPS work and 38% waiting for the next sim frame: the simulator and PyTorch share the GPU.
- **Road rotation is what generalises.** After the first road switch (42k) episode length fell 50 → 23 (−54%); after the switch at ~85k it fell only 46 → 37 (−20%). Training episode length peaked at 62 around 139k.
- **Judge with `--test`, not by watching training.** Training samples random actions and adds jitter, so it always wobbles; the deterministic test is what the car runs.
- **Known gaps:** training runs at about 11 steps/s while the car loop runs at 20 Hz; the model drifts toward the edges and does not always recover; at a road fork it can take the wrong branch. Test throttle is often below 0.1, so it drives cautiously.

### Reproduce

Start the sim and leave it at the main menu; the script then loads the scene itself (it prints `loading scene ...`). `--env_name` has no effect if the sim is already inside a scene.

```
cd project/mycar
python train_sac.py --output sac_a1_baseline                       # train from scratch
python train_sac.py --load <model.zip> --timesteps 50000 --output <new name>   # continue: --timesteps is extra steps
python train_sac.py --test --load models/sac_models/sac_a4_150k.zip --env_name donkey-warren-track-v0 --test_steps 3000
python export_onnx.py --algo sac --model models/sac_models/sac_a4_150k.zip --output sac_a4_150k
```

- The road can be changed mid-run with the sim's exit button; training keeps going and re-sends the camera config.
- On `--load`, the hyperparameters at the top of `train_sac.py` are ignored (stable-baselines3 uses the saved ones); reward and wrapper code changes do apply.
- Checkpoints save the model and its replay buffer every 10k steps. Ctrl+C during training does not save, so stop right after "SAC CHECKPOINT SAVED".

### Files

| Path | What |
|---|---|
| `project/mycar/models/sac_models/sac_a4_150k.zip` | 150k model (stable-baselines3) |
| `project/mycar/models/onnx_models/sac/sac_a4_150k.onnx` | Same model for the car. Input `[1, 1, 40, 160]` uint8, output `[1, 2]` (steering, throttle) |
| `project/mycar/models/sac_models/sac_a*/tb_logs/` | TensorBoard logs for all four runs |

Intermediate checkpoints and replay buffers (25 MB and 257 MB each) are not committed.
