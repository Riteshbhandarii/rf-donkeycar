"""
Export a trained BC / PPO / SAC model to ONNX for lightweight inference.

Usage:
    python export_onnx.py --algo ppo --model models/ppo_models/run1.zip --output run1
    python export_onnx.py --algo sac --model models/sac_models/run1.zip --output run1
    python export_onnx.py --algo bc  --model models/bc_models/run1.pth  --output run1

Output:
    models/onnx_models/<algo>/<output>.onnx
"""

import argparse
import os

import torch
import torch.nn as nn
import numpy as np
import onnxruntime as ort

from stable_baselines3 import PPO, SAC


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(description="Export a trained model to ONNX")
parser.add_argument("--algo", type=str, required=True, choices=["bc", "ppo", "sac"], help="Which kind of model this is.")
parser.add_argument("--model", type=str, required=True, help="Path to the trained model file (.zip for PPO/SAC, .pth/.pt for BC).")
parser.add_argument("--output", type=str, required=True, help="Name for the exported ONNX file (no extension, no path).")
args = parser.parse_args()

ONNX_DIR = os.path.join("models", "onnx_models", args.algo)
os.makedirs(ONNX_DIR, exist_ok=True)
output_path = os.path.join(ONNX_DIR, args.output + ".onnx")


# ============================================================
# DETERMINISTIC ACTOR WRAPPERS
#
# Each wrapper takes the raw (uint8) preprocessed camera image and
# returns a single deterministic action, with SB3's internal
# normalization / squashing / rescaling baked into the graph, so the
# exported ONNX model can be called with nothing but a raw image
# tensor -- no extra pre/post-processing needed at inference time.
# ============================================================

class SACDeterministicWrapper(nn.Module):
    """Wraps an SB3 SAC policy's actor into a plain deterministic module."""

    def __init__(self, policy):
        super().__init__()
        self.actor = policy.actor
        self.low = torch.as_tensor(policy.action_space.low, dtype=torch.float32)
        self.high = torch.as_tensor(policy.action_space.high, dtype=torch.float32)

    def forward(self, obs_uint8):
        obs = obs_uint8.float()
        # SAC's actor returns tanh-squashed actions in [-1, 1] when deterministic=True
        action = self.actor(obs, deterministic=True)
        # Rescale from [-1, 1] to the environment's actual action bounds
        action = self.low + (0.5 * (action + 1.0) * (self.high - self.low))
        return action


class PPODeterministicWrapper(nn.Module):
    """Wraps an SB3 PPO (ActorCriticPolicy) into a plain deterministic module."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        self.low = torch.as_tensor(policy.action_space.low, dtype=torch.float32)
        self.high = torch.as_tensor(policy.action_space.high, dtype=torch.float32)

    def forward(self, obs_uint8):
        obs = obs_uint8.float()
        # Deterministic action = mode of the policy's action distribution
        distribution = self.policy.get_distribution(obs)
        action = distribution.mode()
        # PPO doesn't squash internally -- clip to the environment's action bounds
        action = torch.clamp(action, self.low, self.high)
        return action


class BCModel(nn.Module):
    """
    Must match the architecture in train_bc.py exactly -- this is a plain
    state_dict checkpoint, so PyTorch needs the class definition to load it.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, 50, 160)
            feature_size = self.features(dummy).view(1, -1).shape[1]

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x


class BCWrapper(nn.Module):
    """Wraps the BC model with the same normalization + clipping train_bc.py
    applies at test time, so the exported graph reproduces test-mode behavior
    exactly."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, obs_uint8):
        obs = obs_uint8.float() / 255.0
        action = self.model(obs)
        steering = torch.clamp(action[:, 0:1], -1.0, 1.0)
        throttle = torch.clamp(action[:, 1:2], 0.0, 1.0)
        return torch.cat([steering, throttle], dim=1)


# ============================================================
# LOADERS
# ============================================================

def load_ppo(model_path):
    model = PPO.load(model_path, device="cpu")
    obs_shape = model.policy.observation_space.shape  # (C, H, W)
    wrapper = PPODeterministicWrapper(model.policy)
    return wrapper, obs_shape


def load_sac(model_path):
    model = SAC.load(model_path, device="cpu")
    obs_shape = model.policy.observation_space.shape  # (C, H, W)
    wrapper = SACDeterministicWrapper(model.policy)
    return wrapper, obs_shape


def load_bc(model_path):
    from preprocessing import CROP_TOP

    model = BCModel()
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    img_h, img_w = 120, 160
    obs_shape = (1, img_h - CROP_TOP, img_w)  # (C, H, W) -- matches BCModel's dummy(1,1,50,160)

    wrapper = BCWrapper(model)
    return wrapper, obs_shape


# ============================================================
# EXPORT
# ============================================================

print()
print("===================================")
print("ONNX EXPORT")
print("===================================")
print("Algo:", args.algo)
print("Model:", args.model)
print("Output:", output_path)
print()

if args.algo == "ppo":
    wrapper, obs_shape = load_ppo(args.model)
elif args.algo == "sac":
    wrapper, obs_shape = load_sac(args.model)
else:
    wrapper, obs_shape = load_bc(args.model)

wrapper.eval()

dummy_input = torch.randint(0, 256, (1, *obs_shape), dtype=torch.uint8)

torch.onnx.export(
    wrapper,
    dummy_input,
    output_path,
    input_names=["observation"],
    output_names=["action"],
    opset_version=17,
    dynamic_axes=None,  # fixed batch size of 1 -- matches on-car inference use case
    dynamo=False,
)

print("Exported to:", output_path)

# ============================================================
# VERIFY: compare PyTorch vs ONNX output on the same dummy input
# ============================================================

with torch.no_grad():
    torch_output = wrapper(dummy_input).numpy()

session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
onnx_output = session.run(None, {"observation": dummy_input.numpy()})[0]

max_diff = np.max(np.abs(torch_output - onnx_output))
print("Max abs difference (PyTorch vs ONNX):", max_diff)

if max_diff > 1e-4:
    print("WARNING: output mismatch is larger than expected -- double check the wrapper.")
else:
    print("Outputs match -- export looks correct.")