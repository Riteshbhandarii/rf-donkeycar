import argparse
import sys
import uuid
import os
import numpy as np
import math
import cv2
import torch

from preprocessing import preprocess_frame, CROP_TOP, DebugImageCallback

import gymnasium as gym
import gym_donkeycar  # noqa: F401
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

debugimg = False

# Native macOS: Apple GPU via MPS, sim on the same machine. Docker/lab: CUDA, sim on host.
ON_MAC = sys.platform == "darwin"
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda"
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
parser.add_argument("--timesteps", type=int, default=100000, help="Number of SAC training timesteps.")
parser.add_argument("--load", type=str, default=None, help="Path to existing SAC model to continue training.")
parser.add_argument("--test", action="store_true", help="Load the trained model and drive the simulator.")
parser.add_argument("--test_steps", type=int, default=5000, help="Number of simulator steps during testing.")
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

LEARNING_RATE = 3e-4           # default: 3e-4
BUFFER_SIZE = 40000             # default: 1000000 Insanely large as a default, without any image compressing.
LEARNING_STARTS = 1000          # default: 100
BATCH_SIZE = 128                # default: 256
TAU = 0.005                     # default: 0.005
GAMMA = 0.99                    # default: 0.99
TRAIN_FREQ = 1                  # default: 1
GRADIENT_STEPS = 1              # default: 1
ENT_COEF = "auto"               # default: "auto"
CHECKPOINT_FREQ = 10000         # default: no default

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
    "throttle_min": 0.0,
    "throttle_max": 1.0,

    "cam_config": {
        "img_w": 640,
        "img_h": 480,
        "img_d": 3,
        "fov": 50,
        # Camera position
        "offset_x": 0.0,
        "offset_y": 0.2,
        "offset_z": 1.0,
        "rot_x": -6.0,
    },
}

"""     1step_250.png
        "fov": 50,
        # Camera position
        "offset_x": 0.0,
        "offset_y": 0.0,
        "offset_z": 1.0,
        "rot_x": -8.0,"""

# ============================================================
# IMAGE PREPROCESSING WRAPPERS
# ============================================================

class SupersampleWrapper(gym.ObservationWrapper):
    """Render the sim larger and shrink with area averaging, like the real camera pipeline."""
    def __init__(self, env, out_w=160, out_h=120):
        super().__init__(env)
        self.out_w, self.out_h = out_w, out_h
        self.observation_space = gym.spaces.Box(0, 255, (out_h, out_w, 3), dtype=np.uint8)

    def observation(self, obs):
        if not getattr(self, "_printed", False):
            print("sim frame before resize:", obs.shape)
            self._printed = True
        return cv2.resize(obs, (self.out_w, self.out_h), interpolation=cv2.INTER_AREA)

class PoseJitterWrapper(gym.ObservationWrapper):
    """Training only: per-episode random camera pose change applied to the 160x120 frame.
    pitch -> vertical shift, side shift -> horizontal shift,
    height -> vertical scale about the horizon, FOV -> zoom."""
    def __init__(self, env, fov=50.0, horizon_row=55, pitch_deg=1.0, shift_frac=0.02,
                 height_frac=0.05, fov_deg=3.0):
        super().__init__(env)
        self.h, self.w, _ = env.observation_space.shape
        self.fov, self.horizon_row = fov, horizon_row
        self.pitch_deg, self.shift_frac = pitch_deg, shift_frac
        self.height_frac, self.fov_deg = height_frac, fov_deg
        self.M = np.eye(3)[:2]

    def reset(self, **kwargs):
        rng = np.random.default_rng()
        half = math.radians(self.fov / 2)
        f = (self.h / 2) / math.tan(half)                                      # focal length in pixels
        dy = f * math.tan(math.radians(rng.uniform(-self.pitch_deg, self.pitch_deg)))
        dx = rng.uniform(-self.shift_frac, self.shift_frac) * self.w
        sy = 1.0 + rng.uniform(-self.height_frac, self.height_frac)
        z = math.tan(half) / math.tan(math.radians((self.fov + rng.uniform(-self.fov_deg, self.fov_deg)) / 2))
        cx, cy, yh = self.w / 2, self.h / 2, self.horizon_row
        H = np.array([[1, 0, 0], [0, sy, yh * (1 - sy)], [0, 0, 1]])           # height
        Z = np.array([[z, 0, cx * (1 - z)], [0, z, cy * (1 - z)], [0, 0, 1]])  # zoom
        T = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]])                      # pitch + side shift
        self.M = (T @ Z @ H)[:2]
        return super().reset(**kwargs)

    def observation(self, obs):
        return cv2.warpAffine(obs, self.M, (self.w, self.h),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

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

class AppearanceJitterWrapper(gym.ObservationWrapper):
    """Training only, on the 1-channel observation: per-episode brightness, contrast and gamma,
    sensor noise strength and soft glare spots."""
    def __init__(self, env, contrast=(0.75, 1.25), brightness=20, gamma=(0.8, 1.25),
                 noise_max=4.0, glare_max=2, glare_amp=(20, 60)):
        super().__init__(env)
        self.contrast, self.brightness, self.gamma = contrast, brightness, gamma
        self.noise_max, self.glare_max, self.glare_amp = noise_max, glare_max, glare_amp
        self.h, self.w, _ = env.observation_space.shape
        self.yy, self.xx = np.mgrid[0:self.h, 0:self.w]
        self.rng = np.random.default_rng()
        self._new_episode()

    def _new_episode(self):
        r = self.rng
        self.c = r.uniform(*self.contrast)
        self.b = r.uniform(-self.brightness, self.brightness)
        self.g = r.uniform(*self.gamma)
        self.sigma = r.uniform(0.0, self.noise_max)
        self.glare = [(r.uniform(0, self.w), r.uniform(0, self.h), r.uniform(3, 10), r.uniform(*self.glare_amp))
                      for _ in range(int(r.integers(0, self.glare_max + 1)))]

    def reset(self, **kwargs):
        self._new_episode()
        return super().reset(**kwargs)

    def observation(self, obs):
        x = (obs[..., 0].astype(np.float32) / 255.0) ** self.g
        m = x.mean()
        x = ((x - m) * self.c + m) * 255.0 + self.b
        for cx, cy, s, a in self.glare:
            x += a * np.exp(-((self.xx - cx) ** 2 + (self.yy - cy) ** 2) / (2 * s * s))
        x += self.rng.normal(0.0, self.sigma, x.shape)
        return np.clip(x, 0, 255).astype(np.uint8)[..., np.newaxis]

# ============================================================
# REWARD
# ============================================================

class SimpleRewardWrapper(gym.Wrapper):
    def __init__(self, env, max_cte=6.0, center_sigma=1.1, target_speed=1.5,
                 survival_bonus=0.05, steering_penalty=0.30):
        super().__init__(env)
        self.max_cte, self.center_sigma = max_cte, center_sigma
        self.target_speed = target_speed
        self.survival_bonus, self.steering_penalty = survival_bonus, steering_penalty
        self.prev_steer = 0.0

    def reset(self, **kwargs):
        self.prev_steer = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        cte = info.get("cte", 0.0)
        speed = info.get("forward_vel", 0.0)

        if abs(cte) > self.max_cte:
            return obs, -1.0, True, truncated, info

        center = float(np.exp(-(cte / self.center_sigma) ** 2))
        if speed <= self.target_speed:
            speed_term = speed / self.target_speed
        else:
            speed_term = max(0.0, 1.0 - (speed - self.target_speed) / self.target_speed)
        reward = center * speed_term + self.survival_bonus

        steer = float(action[0])
        reward -= self.steering_penalty * abs(steer - self.prev_steer)
        self.prev_steer = steer
        return obs, reward, terminated, truncated, info

# ============================================================
# ENVIRONMENT
# ============================================================

def make_env():
    env = gym.make(args.env_name, conf=base_conf)
    env = SupersampleWrapper(env)
    if not args.test:
        env = PoseJitterWrapper(env, fov=base_conf["cam_config"]["fov"])
    env = CropGrayscaleWrapper(env)
    if not args.test:
        env = AppearanceJitterWrapper(env)
    env = SimpleRewardWrapper(env)
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
# CODE LOG CALLBACK
# ============================================================

class CodeLogCallback(BaseCallback):
    """Writes this script to TensorBoard (TEXT tab) at the start of the run."""
    def _on_training_start(self) -> None:
        with open(os.path.abspath(__file__)) as f:
            text = "\n".join("    " + line for line in f.read().splitlines())   # 4 spaces = code block
        self.logger.record("code/train_sac.py", text, exclude=("stdout", "log", "json", "csv"))
        self.logger.dump(self.num_timesteps)

    def _on_step(self) -> bool:
        return True

# ============================================================
# TEST MODE
# ============================================================

if args.test:
    model_path = args.load or output_path
    print()
    print("===================================")
    print("SAC MODEL TEST")
    print("===================================")
    print("Model:", model_path)
    print("Environment:", args.env_name)
    print("Port:", args.port)
    print("Steps:", args.test_steps)
    print()

    model = SAC.load(model_path, env=env, device=DEVICE)
    obs = env.reset()

    step, crashes, laps, ep_laps = -1, 0, 0, 0
    try:
        for step in range(args.test_steps):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_laps = info[0].get("lap_count", 0)

            if step % 20 == 0:
                steering = float(action[0][0])
                throttle = float(action[0][1])
                print(f"step={step:5d} | steering={steering:+.3f} | throttle={throttle:.3f}")

            if done[0]:
                crashes += 1
                laps += ep_laps
                ep_laps = 0
                print(f"step={step:5d} | episode ended (crash or off track)")
                obs = env.reset()

    except KeyboardInterrupt:
        print("\nTest interrupted.")
    finally:
        env.close()

    laps += ep_laps
    print(f"Test finished. Steps: {step + 1} | crashes: {crashes} | laps: {laps}")
    exit(0)

# ============================================================
# MODEL
# ============================================================

if args.load:
    print()
    print("Loading SAC model:")
    print(args.load)

    model = SAC.load(args.load, env=env, device=DEVICE, tensorboard_log=tensorboard_dir)

    buffer_path = args.load.replace(".zip", "_buffer.pkl")

    print("Loading replay buffer:")
    print(buffer_path)

    model.load_replay_buffer(buffer_path)
    print("Continuing SAC training...")

else:
    print()
    print("Creating new SAC model...")
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], qf=[256, 256]))
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
callbacks.append(CodeLogCallback())

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
print("Timesteps:", args.timesteps)
print("Buffer size:", BUFFER_SIZE)
print("Batch size:", BATCH_SIZE)
print("Learning rate:", LEARNING_RATE)
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