"""
Shared image preprocessing for the RC car RL pipeline.

Used by BOTH:
  - training (train_sac.py, via CropGrayscaleWrapper)
  - car inference (ONNX runtime script on the Jetson)

"""

import numpy as np

# ==============================================================
# CONFIG
# ==============================================================

# Rows to remove from the top of the image (sky / horizon / background).
CROP_TOP = 70

# Grayscale channel weights (standard luma / ITU-R BT.601 weights,
# same ones OpenCV's cv2.cvtColor(..., COLOR_RGB2GRAY) uses).
GRAY_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)


# ==============================================================
# CORE TRANSFORM
# ==============================================================

def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    frame: HxWx3 uint8, RGB order.
    returns: (H - CROP_TOP) x W x 1 uint8, grayscale.
    """
    cropped = frame[CROP_TOP:, :, :]
    gray = np.dot(cropped[..., :3].astype(np.float32), GRAY_WEIGHTS)
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray[..., np.newaxis]


def output_shape(input_h: int, input_w: int) -> tuple:
    """Given the raw camera/sim frame H,W, return the shape after preprocessing."""
    return (input_h - CROP_TOP, input_w, 1)


# ==============================================================
# DEBUG CALLBACK (SB3)
# ==============================================================
# Saves the current (preprocessed) observation as a PNG every
# `save_freq` timesteps, so you can visually confirm crop/grayscale
# is doing what you expect during a real training run.
#
# Usage (in train_sac.py / train_ppo.py):
#     from preprocessing import DebugImageCallback
#     debug_callback = DebugImageCallback(save_freq=1000, save_path="debug_images")
#     model.learn(..., callback=[checkpoint_callback, debug_callback])

from pathlib import Path
import cv2
from stable_baselines3.common.callbacks import BaseCallback


class DebugImageCallback(BaseCallback):
    def __init__(self, save_freq=1000, save_path="debug_images", verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            # new_obs shape: (n_envs, C, H, W) after VecTransposeImage
            obs = self.locals["new_obs"][0]        # first env
            img = np.transpose(obs, (1, 2, 0))     # CHW -> HWC
            if img.shape[2] == 1:
                img = img[:, :, 0]                 # drop channel dim for grayscale

            filename = self.save_path / f"step_{self.num_timesteps}.png"
            cv2.imwrite(str(filename), img)

            if self.verbose:
                print(f"[DebugImageCallback] saved {filename}")

        return True