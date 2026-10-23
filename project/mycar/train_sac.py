import argparse
import sys
import uuid
import os
import numpy as np

from preprocessing import preprocess_frame, CROP_TOP, DebugImageCallback

import torch
import gymnasium as gym
import gym_donkeycar  # noqa: F401
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

debugimg = False

# Native macOS: Apple GPU via MPS, sim on the same machine. Docker/lab: CUDA, sim on host.
ON_MAC = sys.platform == "darwin"
DEVICE = "mps" if torch.backends.mps.is_available() else "auto"
SIM_HOST = "127.0.0.1" if ON_MAC else "host.docker.internal"

# ============================================================
# ENVIRONMENTS
# ============================================================

env_list = [
    "donkey-generated-roads-v0",
    "donkey-warren-track-v0",
]

# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(description="SAC training for DonkeyCar")
parser.add_argument("--sim", type=str, default="manual", help="Path to Unity simulator. Use 'manual' when starting simulator yourself.")
parser.add_argument("--port", type=int, default=9091, help="DonkeySim TCP port.")
parser.add_argument("--env_name", type=str, default="donkey-generated-roads-v0", choices=env_list, help="DonkeySim environment.")
parser.add_argument("--timesteps", type=int, default=None, help="Number of SAC training timesteps. Default 100000, or 30000 with --fast.")
parser.add_argument("--load", type=str, default=None, help="Path to existing SAC model to continue training.")
parser.add_argument("--test", action="store_true", help="Load the trained model and drive the simulator.")
parser.add_argument("--test_steps", type=int, default=5000, help="Number of simulator steps during testing.")
parser.add_argument("--reward", type=str, default="simple", choices=["simple", "smooth", "survival"], help="Reward: 'simple' (instructor default), 'smooth' (adds an anti-stall penalty) or 'survival' (araffin: +1 per step alive, -10 crash).")
parser.add_argument("--fast", action="store_true", help="Mac speed mode: half the gradient updates, smaller batches, shorter run, checkpoints every 5000 steps.")
parser.add_argument("--zoo", action="store_true", help="araffin's tuned DonkeyCar SAC/TQC settings (RL Zoo): lr 7.3e-4, tau 0.02, batch 256, gSDE, throttle 0.4-0.6. Fresh runs only.")
parser.add_argument("--lr", type=float, default=None, help="Override learning rate (default: script value, or the checkpoint's with --load).")
parser.add_argument("--gamma", type=float, default=None, help="Override discount factor (default: script value, or the checkpoint's with --load).")
parser.add_argument("--tau", type=float, default=None, help="Override target-network update rate (default: script value, or the checkpoint's with --load).")
parser.add_argument("--output", type=str, default="sac_donkey", help="Path/name for final SAC model.(saved under models/sac_models/)")
args = parser.parse_args()

# Making directories
MODEL_DIR = "models/sac_models"
output_path = os.path.join(MODEL_DIR, args.output)
checkpoint_dir = os.path.join(output_path, "checkpoints")
tensorboard_dir = os.path.join(output_path, "tb_logs")

os.makedirs(output_path, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(tensorboard_dir, exist_ok=True)

#tensorboard --logdir models/sac_models/ --bind_all
#http://localhost:6006/

# ============================================================
# SAC HYPERPARAMETERS
# ============================================================

LEARNING_RATE = 3e-4            # default: 3e-4
BUFFER_SIZE = 40000             # default: 1000000 Insanely large as a default, without any image compressing.
LEARNING_STARTS = 1000          # default: 100
BATCH_SIZE = 128                # default: 256
TAU = 0.005                     # default: 0.005
GAMMA = 0.99                    # default: 0.99
TRAIN_FREQ = 1                  # default: 1
GRADIENT_STEPS = 1              # default: 1
ENT_COEF = "auto"               # default: "auto"
USE_SDE = False                 # default: False
SDE_SAMPLE_FREQ = -1            # default: -1
LOG_STD_INIT = -3               # default: -3, only used with gSDE
THROTTLE_MIN = 0.0
THROTTLE_MAX = 1.0
CHECKPOINT_FREQ = 25000         # default: no default

# --fast: on a Mac (MPS) one SAC update costs ~100 ms, so updating on every step caps
# training at ~5 steps/s. Updating twice every 4 steps with smaller batches roughly
# doubles that, and a 30k-step run gives a first answer in under an hour.
if args.fast:
    BATCH_SIZE = 64
    TRAIN_FREQ = 4
    GRADIENT_STEPS = 2
    CHECKPOINT_FREQ = 5000

# --zoo: araffin's tuned DonkeyCar settings (TQC model card on Hugging Face, araffin/tqc-donkey-minimonaco-track-v0,
# and learning-to-drive-in-5-minutes config.py). Throttle capped so the car is slow enough to learn steering.
# Buffer 100k instead of 200k: 50x160 grayscale frames, ~1.6 GB of RAM.
if args.zoo:
    LEARNING_RATE = 7.3e-4
    BUFFER_SIZE = 100000
    LEARNING_STARTS = 500
    BATCH_SIZE = 256
    TAU = 0.02
    GAMMA = 0.99
    USE_SDE = True
    SDE_SAMPLE_FREQ = 16
    THROTTLE_MIN = 0.4
    THROTTLE_MAX = 0.6
    # Train after each episode instead of every step, as many updates as steps collected.
    # Per-step updates (~100 ms on MPS) drop the control rate to ~4 fps and the car drives blind.
    TRAIN_FREQ = (1, "episode")
    GRADIENT_STEPS = -1
    CHECKPOINT_FREQ = 5000

# Fine-tuning overrides (fresh runs). With --load only the flags you pass are applied, see MODEL.
if args.lr is not None:
    LEARNING_RATE = args.lr
if args.gamma is not None:
    GAMMA = args.gamma
if args.tau is not None:
    TAU = args.tau

if args.timesteps is None:
    args.timesteps = 30000 if args.fast else 100000

# ============================================================
# DONKEY SIM CONFIGURATION
# ============================================================

base_conf = {
    "exe_path": args.sim,
    "port": args.port,
    "bio": "SAC Learning to Drive",
    "guid": str(uuid.uuid4()),
    "host": SIM_HOST,
    "body_style": "f1",
    "body_rgb": (255, 0, 0),
    "car_name": "SAC_Agent",
    "font_size": 10,
    "max_cte": 10.0,
    "frame_skip": 1,
    "throttle_min": THROTTLE_MIN,
    "throttle_max": THROTTLE_MAX,

    "cam_config": {
        "img_w": 160,
        "img_h": 120,
        "img_d": 3,
        # Camera position
        "offset_x": 0.0,
        "offset_y": 0.7,
        "offset_z": 1.0,
        "rot_x": -19.0,
    },
}


# ============================================================
# IMAGE PREPROCESSING WRAPPER
# ============================================================

class CropGrayscaleWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        h, w, _ = env.observation_space.shape
        new_h = h - CROP_TOP
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(new_h, w, 1), dtype=np.uint8
        )

    def observation(self, obs):
        return preprocess_frame(obs)

# ============================================================
# REWARD
# ============================================================

class SimpleRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        self.prev_steer = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        cte = info.get("cte", 0.0)
        speed = info.get("forward_vel", 0.0)

        # Reward for staying near the center of the track
        max_cte = 7.0

        # Off track
        if abs(cte) > max_cte:
            terminated = True
            reward = -1.0

        # Moving forward
        elif speed > 0.0:
            reward = (1.0 - abs(cte) / max_cte) * speed
        # Not moving
        else:
            reward = 0.0
        
        # Steering smoothness
        steer_delta = abs(action[0] - self.prev_steer)
        reward -= 0.1 * steer_delta
        self.prev_steer = action[0]

        # Survival reward for maintaining forward movement
        if speed >= 0.4:
            reward += 0.2

        return obs, reward, terminated, truncated, info

# Anti-stall variant (opt-in with --reward smooth). Same shape as SimpleRewardWrapper, but:
# - standing still while on track is penalised, so "do nothing" is no longer a safe
#   optimum (otherwise the deterministic --test policy can settle at ~0 throttle and
#   only the random exploration throttle moves the car in training)
# - the survival bonus is only paid when moving fast enough AND near the centre line
# - steering smoothness is kept (limits wiggling on the centre line)
class SmoothRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_steer = 0.0

    def reset(self, **kwargs):
        self.prev_steer = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        cte = info.get("cte", 0.0)
        speed = info.get("forward_vel", 0.0)
        steer = float(action[0])

        max_cte = 7.0
        min_speed = 0.1         # below this the car counts as stalled

        # Off track
        if abs(cte) > max_cte:
            terminated = True
            reward = -1.0

        # Moving forward
        elif speed > min_speed:
            reward = (1.0 - abs(cte) / max_cte) * speed
        # Not moving: small penalty instead of 0
        else:
            reward = -0.1

        # Steering smoothness
        reward -= 0.1 * abs(steer - self.prev_steer)
        self.prev_steer = steer

        # Survival reward only for real forward movement close to the centre line
        if speed >= 0.4 and abs(cte) < 0.5 * max_cte:
            reward += 0.2

        return obs, float(reward), terminated, truncated, info

# araffin's reward (learning-to-drive-in-5-minutes): +1 for every step alive plus a small
# throttle bonus, crash penalty grows with the throttle at the moment of the crash.
class SurvivalRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        cte = info.get("cte", 0.0)
        throttle = float(action[1])

        max_cte = 7.0

        # Off track
        if abs(cte) > max_cte:
            terminated = True

        # Crash: -10, minus up to 5 more at full throttle
        if terminated:
            norm_throttle = (throttle - THROTTLE_MIN) / max(THROTTLE_MAX - THROTTLE_MIN, 1e-6)
            reward = -10.0 - 5.0 * norm_throttle
        # Alive: +1 per step, plus throttle bonus
        else:
            reward = 1.0 + 0.1 * throttle / THROTTLE_MAX

        return obs, float(reward), terminated, truncated, info

REWARD_WRAPPERS = {"simple": SimpleRewardWrapper, "smooth": SmoothRewardWrapper, "survival": SurvivalRewardWrapper}

# ============================================================
# ENVIRONMENT
# ============================================================

def make_env():
    env = gym.make(args.env_name, conf=base_conf)
    env = CropGrayscaleWrapper(env)
    env = REWARD_WRAPPERS[args.reward](env)
    env = Monitor(env)
    return env


env = DummyVecEnv([make_env])
env = VecTransposeImage(env)

# ============================================================
# SAC CHECKPOINT CALLBACK
# ============================================================

class SACCheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path, name_prefix="sac_donkey", verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _on_step(self):
        if self.num_timesteps % self.save_freq == 0:

            model_path = (
                f"{self.save_path}/"
                f"{self.name_prefix}_{self.num_timesteps}_steps"
            )

            buffer_path = model_path + "_buffer.pkl"

            self.model.save(model_path)
            self.model.save_replay_buffer(buffer_path)

            if self.verbose:
                print()
                print("===================================")
                print("SAC CHECKPOINT SAVED")
                print("===================================")
                print("Model:", model_path + ".zip")
                print("Replay buffer:", buffer_path)
                print()

        return True

# ============================================================
# TEST MODE
# ============================================================

# SB3 accepts a model path with or without ".zip"; the replay buffer sits next to it.
# (The default output_path is also a directory, the model itself is output_path + ".zip".)
def with_zip(path):
    return path if path.endswith(".zip") else path + ".zip"

if args.test:
    model_path = with_zip(args.load or output_path)
    print()
    print("===================================")
    print("SAC MODEL TEST")
    print("===================================")
    print("Model:", model_path)
    print("Environment:", args.env_name)
    print("Reward:", args.reward)
    print("Port:", args.port)
    print("Steps:", args.test_steps)
    print()

    model = SAC.load(model_path, env=env, device=DEVICE)
    obs = env.reset()

    try:
        for step in range(args.test_steps):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            if step % 20 == 0:
                steering = float(action[0][0])
                throttle = float(action[0][1])
                print(f"step={step:5d} | steering={steering:+.3f} | throttle={throttle:.3f}")

            if done[0]:
                obs = env.reset()

    except KeyboardInterrupt:
        print("\nTest interrupted.")
    finally:
        env.close()

    print("Test finished.")
    exit(0)

# ============================================================
# MODEL
# ============================================================

if args.load:
    print()
    print("Loading SAC model:")
    print(args.load)

    # --fast overrides the update schedule saved in the model
    fast_kwargs = dict(batch_size=BATCH_SIZE, train_freq=TRAIN_FREQ, gradient_steps=GRADIENT_STEPS) if args.fast else {}
    # --lr/--gamma/--tau: safe to change when continuing; SAC.load rebuilds lr_schedule from them
    # and train() pushes the schedule into the optimizers, so they take effect on the first update
    tune_kwargs = {k: v for k, v in dict(learning_rate=args.lr, gamma=args.gamma, tau=args.tau).items() if v is not None}
    model = SAC.load(args.load, env=env, device=DEVICE, tensorboard_log=tensorboard_dir, **fast_kwargs, **tune_kwargs)

    buffer_path = with_zip(args.load)[:-len(".zip")] + "_buffer.pkl"

    if os.path.isfile(buffer_path):
        print("Loading replay buffer:")
        print(buffer_path)
        model.load_replay_buffer(buffer_path)
    else:
        # Empty buffer: wait for LEARNING_STARTS new steps before training again,
        # otherwise updates would sample from an almost empty buffer.
        print("WARNING: replay buffer not found:", buffer_path)
        print("Continuing with an EMPTY replay buffer (collecting", LEARNING_STARTS, "new steps first).")
        model.learning_starts = model.num_timesteps + LEARNING_STARTS
    print("Continuing SAC training...")

else:
    print()
    print("Creating new SAC model...")
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], qf=[256, 256]))
    if USE_SDE:
        policy_kwargs.update(log_std_init=LOG_STD_INIT, use_expln=True)
    model = SAC("CnnPolicy",
        env, 
        learning_rate=LEARNING_RATE, 
        buffer_size=BUFFER_SIZE,
        learning_starts=LEARNING_STARTS, 
        batch_size=BATCH_SIZE, 
        tau=TAU,
        gamma=GAMMA, 
        train_freq=TRAIN_FREQ, 
        gradient_steps=GRADIENT_STEPS,
        ent_coef=ENT_COEF,
        use_sde=USE_SDE,
        sde_sample_freq=SDE_SAMPLE_FREQ,
        use_sde_at_warmup=USE_SDE,
        policy_kwargs=policy_kwargs,
        optimize_memory_usage=True, 
        replay_buffer_kwargs=dict(handle_timeout_termination=False),
        verbose=1, 
        device=DEVICE,
        tensorboard_log=tensorboard_dir
        )

# ============================================================
# CHECKPOINTS
# ============================================================

checkpoint_callback = SACCheckpointCallback(
    save_freq=CHECKPOINT_FREQ, 
    save_path=checkpoint_dir, 
    name_prefix="sac_donkey"
    )

class SpeedLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_speeds = None

    def _on_training_start(self) -> None:
        n_envs = self.training_env.num_envs
        self.episode_speeds = [[] for _ in range(n_envs)]

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        dones = self.locals["dones"]

        for i, info in enumerate(infos):
            self.episode_speeds[i].append(info.get("forward_vel", 0.0))

            if dones[i]:
                speeds = self.episode_speeds[i]
                if speeds:
                    self.logger.record("rollout/avg_speed", sum(speeds) / len(speeds))
                self.episode_speeds[i] = []

        return True
    
speed_callback = SpeedLoggingCallback()
callbacks = [checkpoint_callback, speed_callback]

# --zoo trains between episodes. The sim keeps applying the last control command,
# so hold the reset car with the hand brake until the next action releases it.
class BrakeDuringTrainingCallback(BaseCallback):
    def _on_rollout_end(self) -> None:
        for env in self.training_env.envs:
            env.unwrapped.viewer.handler.send_control(0, 0, 1.0)

    def _on_step(self) -> bool:
        return True

if args.zoo:
    callbacks.append(BrakeDuringTrainingCallback())

# Debug image
if debugimg:
    debug_callback = DebugImageCallback(save_freq=50, save_path="debug_images")
    callbacks.append(debug_callback)
    print("Debug image callback enabled to debug_images/")

# ============================================================
# TRAINING
# ============================================================

print()
print("===================================")
print("SAC TRAINING")
print("===================================")
print("Environment:", args.env_name)
print("Reward:", args.reward)
print("Timesteps:", args.timesteps)
print("Buffer size:", BUFFER_SIZE)
print("Batch size:", BATCH_SIZE)
print("Updates:", GRADIENT_STEPS, "every", TRAIN_FREQ, "steps", "(--fast)" if args.fast else "")
print("Learning rate:", model.lr_schedule(1.0))
print("Gamma:", model.gamma)
print("Tau:", model.tau)
print("gSDE:", model.use_sde, "| Throttle:", THROTTLE_MIN, "-", THROTTLE_MAX, "(--zoo)" if args.zoo else "")
print("Learning starts:", LEARNING_STARTS)
print("Output:", args.output)
print()


model.learn(
    total_timesteps=args.timesteps,
    callback=callbacks,
    progress_bar=True,
    reset_num_timesteps=False,
    log_interval=10,
    tb_log_name=args.output
)

# ============================================================
# SAVE MODEL
# ============================================================

model.save(output_path)

buffer_path = output_path + "_buffer.pkl"
model.save_replay_buffer(buffer_path)

print()
print("===================================")
print("TRAINING COMPLETE")
print("===================================")
print("Model saved to:", output_path + ".zip")
print("Replay buffer saved to:", buffer_path)

# ============================================================
# CLOSE
# ============================================================

env.close()