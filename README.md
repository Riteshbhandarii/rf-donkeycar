# rf-donkeycar

Self-driving experiments for DonkeyCar in the Donkey simulator (`gym_donkeycar`).

## What's here

| Path | What it is |
|---|---|
| `project/mycar/train_ppo.py` | PPO training and testing with stable-baselines3 |
| `train_bc.py` | Behavioural cloning training in PyTorch |
| `project/mycar/manage.py`, `train.py`, `calibrate.py`, `config.py`, `myconfig.py` | Standard DonkeyCar car app scripts and config |
| `preprocessing.py` | Frame preprocessing shared by the training scripts |
| `project/mycar/models/ppo_models/ppo_session3/checkpoints/` | PPO checkpoints at 20k and 40k steps |
| `project/mycar/models/ppo_models/*/tb_logs/` | TensorBoard logs |
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
