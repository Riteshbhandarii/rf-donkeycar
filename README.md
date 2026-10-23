# rf-donkeycar

Self-driving experiments for DonkeyCar in the Donkey simulator (`gym_donkeycar`).

## What's here

| Path | What it is |
|---|---|
| `project/mycar/train_ppo.py` | PPO training and testing with stable-baselines3 |
| `project/mycar/train_sac.py` | SAC training and testing with stable-baselines3 (Session 4) |
| `project/mycar/compare_runs.py` | Plots reward, episode length and speed of PPO/SAC runs on one PNG |
| `train_bc.py` | Behavioural cloning training in PyTorch |
| `project/mycar/manage.py`, `train.py`, `calibrate.py`, `config.py`, `myconfig.py` | Standard DonkeyCar car app scripts and config |
| `preprocessing.py` | Frame preprocessing shared by the training scripts |
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

Reward: `train_sac.py --reward smooth` targets the "drives in training, stands still in --test" problem: stalling on track costs -0.1 instead of 0, and the +0.2 survival bonus only pays when moving near the centre line. Steering smoothness is in both rewards. It is opt-in; the default reward is the instructor's, unchanged.
