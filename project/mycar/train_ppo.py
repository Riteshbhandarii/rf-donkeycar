import argparse
import os
import sys
import uuid

import gymnasium as gym
import gym_donkeycar  # noqa: F401
import numpy as np

from preprocessing import preprocess_frame, CROP_TOP, DebugImageCallback

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
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

parser = argparse.ArgumentParser(description="PPO training for DonkeyCar")
parser.add_argument("--sim", type=str, default="manual", help="Path to Unity simulator. Use 'manual' when starting simulator yourself.")
parser.add_argument("--port", type=int, default=9091, help="DonkeySim TCP port.")
parser.add_argument("--env_name", type=str, default="donkey-generated-roads-v0", choices=env_list, help="DonkeySim environment.")
parser.add_argument("--timesteps", type=int, default=100000, help="Number of PPO training timesteps.")
parser.add_argument("--load", type=str, default=None, help="Path to existing PPO model to continue training.")
parser.add_argument("--test", action="store_true", help="Load the trained model and drive the simulator.")
parser.add_argument("--test_steps", type=int, default=5000, help="Number of simulator steps during testing.")
parser.add_argument("--output", type=str, default="ppo_donkey", help="Path/name for final PPO model.(saved under models/ppo_models/)")
args = parser.parse_args()

MODEL_DIR = "models/ppo_models"
output_path = os.path.join(MODEL_DIR, args.output)
checkpoint_dir = os.path.join(output_path, "checkpoints")
TENSORBOARD_LOG_DIR = os.path.join(output_path, "tb_logs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(output_path, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)

#tensorboard --logdir models/ppo_models/ --bind_all
#http://localhost:6006/

# ============================================================
# PPO HYPERPARAMETERS
# ============================================================

N_STEPS = 2048                      # default: 2048
BATCH_SIZE = 64                     # default: 64
LEARNING_RATE = 3e-4                # default: 3e-4
GAMMA = 0.99                        # default: 0.99
GAE_LAMBDA = 0.95                   # default: 0.95
CLIP_RANGE = 0.2                    # default: 0.2
ENT_COEF = 0.0                      # default: 0.0
N_EPOCHS = 10                       # default: 10
CHECKPOINT_FREQ = 20000
TARGET_KL = 0.05                    # default: None

# ============================================================
# DONKEY SIM CONFIGURATION
# ============================================================

base_conf = {
    "exe_path": args.sim,
    "port": args.port,
    "bio": "PPO Learning to Drive",
    "guid": str(uuid.uuid4()),
    "host": SIM_HOST,
    "body_style": "f1",
    "body_rgb": (255, 0, 0),
    "car_name": "PPO_Agent",
    "font_size": 10,
    "max_cte": 10.0,
    "frame_skip": 1,
    "throttle_min": 0.0,
    "throttle_max": 1.0,

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
# REWARDSYSTEM
# ============================================================

class SimpleRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        #print("INFO:", info)   #
        cte = info.get("cte", 0.0)
        speed = info.get("forward_vel", 0.0)

        # Maximum distance from the center of track
        max_cte = 7.0
        max_speed = 1.8

        centering = max(1.0 - abs(cte) / max_cte, 0.0)
        speed_factor = np.clip(speed / max_speed, 0.0, 1.0)

        reward = 2.0 * centering * speed_factor

        if abs(cte) > max_cte:
            terminated = True
            reward = -2.0

        return obs, reward, terminated, truncated, info

# ============================================================
# ENVIRONMENT
# ============================================================

def make_env():
    env = gym.make(args.env_name, conf=base_conf)
    env = CropGrayscaleWrapper(env)
    env = SimpleRewardWrapper(env)
    env = Monitor(env)
    return env


env = DummyVecEnv([make_env])
env = VecTransposeImage(env)

# ============================================================
# TEST MODE
# ============================================================

if args.test:
    model_path = args.load or output_path

    print()
    print("===================================")
    print("PPO MODEL TEST")
    print("===================================")
    print("Model:", model_path)
    print("Environment:", args.env_name)
    print("Port:", args.port)
    print("Steps:", args.test_steps)
    print()

    model = PPO.load(model_path, env=env, device=DEVICE)
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
    print("Loading PPO model:")
    print(args.load)

    model = PPO.load(args.load, env=env, device=DEVICE, tensorboard_log=TENSORBOARD_LOG_DIR)
    print("Continuing PPO training...")

else:
    print()
    print("Creating new PPO model...")
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
    model = PPO("CnnPolicy", env, learning_rate=LEARNING_RATE, n_steps=N_STEPS, 
                batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, gamma=GAMMA, gae_lambda=GAE_LAMBDA,
                clip_range=CLIP_RANGE, target_kl=TARGET_KL, ent_coef=ENT_COEF, 
                policy_kwargs=policy_kwargs, verbose=1, device=DEVICE, 
                tensorboard_log=TENSORBOARD_LOG_DIR)

# ============================================================
# CHECKPOINTS / CALLBACKS
# ============================================================

checkpoint_callback = CheckpointCallback(
    save_freq=CHECKPOINT_FREQ, 
    save_path=checkpoint_dir, 
    name_prefix="ppo_donkey"
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
print("PPO TRAINING")
print("===================================")
print("Environment:", args.env_name)
print("Timesteps:", args.timesteps)
print("N steps:", N_STEPS)
print("Batch size:", BATCH_SIZE)
print("Learning rate:", LEARNING_RATE)
print("Output:", args.output)
print()


model.learn(
    total_timesteps=args.timesteps,
    callback=callbacks,
    progress_bar=True,
    reset_num_timesteps=False,
    log_interval=1,
    tb_log_name=args.output,
)

model.save(output_path)

print()
print("===================================")
print("TRAINING COMPLETE")
print("===================================")
print("Model saved to:", output_path + ".zip")


# ============================================================
# CLOSE
# ============================================================

env.close()