import argparse
import copy
import json
from pathlib import Path

import cv2
import gymnasium as gym
import gym_donkeycar  # noqa: F401
import numpy as np

from preprocessing import preprocess_frame, CROP_TOP

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


# ============================================================
# ENVIRONMENTS
# ============================================================

# Does only matter if you run the script when menu is open in simulator
env_list = [
    "donkey-generated-roads-v0",
    "donkey-warren-track-v0",
]


# ============================================================
# ARGUMENTS
# ============================================================

# Arguments that you can use when running the script. For example:
# python train_bc.py --tubs data/tub_1 data/tub_2 --epochs 100 --batch_size 32 --lr 0.001 --output models/bc_models/HamiltonCloner.pth

parser = argparse.ArgumentParser(description="PyTorch Behavior Cloning for DonkeyCar")
parser.add_argument("--tubs", type=str, default=None, help="Path to tub directory. Multiple tubs separated by comma.") # data/tub_1_26-08-21
parser.add_argument("--model", type=str, default=None, help="Path to existing .pth model to load.")
parser.add_argument("--output", type=str, default="models/bc_models/bc_model.pth", help="Path for best model.")
parser.add_argument("--epochs", type=int, default=50, help="Maximum number of training epochs.")
parser.add_argument("--batch_size", type=int, default=64, help="Training batch size.")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
parser.add_argument("--patience", type=int, default=10, help="Early stopping patience.")
parser.add_argument("--num_workers", type=int, default=2, help="Number of DataLoader workers.")
parser.add_argument("--port", type=int, default=9091, help="DonkeySim TCP port.")
parser.add_argument("--env_name", type=str, default="donkey-generated-roads-v0", choices=env_list, help="DonkeySim environment used for testing.")
parser.add_argument("--test", action="store_true", help="Load the trained model and drive the simulator.")
parser.add_argument("--test_steps", type=int, default=5000, help="Number of simulator steps during testing.")
args = parser.parse_args()

# Many tubs --tubs data/tub1 data/tub2 data/tub3

# ============================================================
# DEVICE
# ============================================================

# Makes sure that the model is trained on GPU if available, otherwise CPU.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")           # CUDA is Nvidias GPU, if you have one, it will use it
print("Device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# DONKEY SIM CONFIGURATION
# ============================================================

# Configs for the DonkeySim environment. You can change these values to customize the simulator settings.

base_conf = {
    "exe_path": "manual",
    "port": args.port,
    "bio": "Behavior Cloning",
    "host": "host.docker.internal",
    "body_style": "f1",
    "body_rgb": (255, 0, 0),
    "car_name": "BC_Agent",
    "font_size": 10,
    "max_cte": 10.0,
    "frame_skip": 1,
    "throttle_min": 0.0,
    "throttle_max": 1.0,
    "cam_config": {
        "img_w": 160,
        "img_h": 120,
        "img_d": 3,
        "offset_x": 0.0,
        "offset_y": 0.0,
        "offset_z": 1.0,
        "rot_x": -19.0,
    },
}


# ============================================================
# IMAGE PREPROCESSING WRAPPER
# ============================================================

# Adds Cropping and Grayscale preprocessing to the DonkeySim environment.

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
# DATASET
# ============================================================

# Dataset class for loading DonkeyCar tub data and counts number of samples
# Reads the catalog files in the specified tub directories, loads the images and corresponding steering/throttle values.
# Applies preprocessing to the images.

class DonkeyBCDataset(Dataset):
    def __init__(self, tub_paths):
        self.samples = []
        self.tub_boundaries = []
        for tub_path in tub_paths:
            tub_path = Path(tub_path)
            if not tub_path.exists():
                raise FileNotFoundError(f"Tub not found: {tub_path}")
            print(f"Loading tub: {tub_path}")

            start_idx = len(self.samples)

            catalog_files = sorted(tub_path.glob("catalog_*.catalog"))
            if not catalog_files:
                raise RuntimeError(f"No catalog files found in {tub_path}")

            for catalog_file in catalog_files:
                with open(catalog_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue

                        record = json.loads(line)
                        image_name = record.get("cam/image_array")
                        steering = record.get("user/angle")
                        throttle = record.get("user/throttle")

                        if (image_name is None or steering is None or throttle is None):
                            continue

                        image_path = tub_path / "images" / image_name

                        if not image_path.exists():
                            continue

                        self.samples.append((str(image_path), float(steering), float(throttle)))

            self.tub_boundaries.append((start_idx, len(self.samples)))

        print(f"Total samples: {len(self.samples)}")

        if len(self.samples) == 0:
            raise RuntimeError("No valid training samples found.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, steering, throttle = self.samples[index]
        image = cv2.imread(image_path)

        if image is None:
            raise RuntimeError(
                f"Could not read image: {image_path}"
            )

        # OpenCV BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)          # OpenCV and Neural Networks use different color formats
        # Crop + grayscale (same transform used in training and on the car)
        image = preprocess_frame(image)                         # HxWx1 uint8
        # Convert uint8 -> float32
        image = image.astype(np.float32) / 255.0                # Camera image is in uint8 [0, 255], we need to convert it to float32 and normalize it to [0, 1]
        # HWC -> CHW
        image = np.transpose(image, (2, 0, 1))                  # Now (1, H, W) for grayscale
        image = torch.from_numpy(image)
        target = torch.tensor([steering, throttle], dtype=torch.float32)
        return image, target


# ============================================================
# CNN MODEL
# ============================================================

# CNN model for behavior cloning.
# Takes a preprocessed grayscale image as input and outputs steering and throttle values.

class BCModel(nn.Module):
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

        # Determine flattened size automatically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 50, 160)                  # Input is 120x160, cropped by 70 pixels -> 50x160
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


# ============================================================
# MODEL
# ============================================================

# Initializes the BC model and loads a pre-trained model if specified.

model = BCModel().to(device)

if args.model is not None:
    print(f"Loading model from: {args.model}")
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint)
    print("Model loaded.")


# ============================================================
# TEST MODE
# ============================================================

# If the --test argument is set, the script will run the trained model.

if args.test:
    model.eval()
    env = gym.make(args.env_name, conf=base_conf)
    env = CropGrayscaleWrapper(env)
    obs, info = env.reset()
    """
    debug_dir = Path("debug_images")                                # Debug_images
    debug_dir.mkdir(parents=True, exist_ok=True)                    # Depug_Images
    """
    print()
    print("===================================")
    print("BC MODEL TEST")
    print("===================================")
    print("Environment:", args.env_name)
    print("Port:", args.port)
    print("Steps:", args.test_steps)
    print()
    print("Drive the simulator using the model.")
    print("Press Ctrl+C to stop.")
    print()

    try:
        for step in range(args.test_steps):
            image = obs
            """
            Debug_images Save what the model actually sees, after crop+grayscale, before normalization
            if step % 20 == 0:
                debug_frame = obs.squeeze(-1)  # (H, W, 1) -> (H, W), uint8
                cv2.imwrite(str(debug_dir / f"step_{step:05d}.png"), debug_frame)
            """
            image = image.astype(np.float32) / 255.0
            image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image).unsqueeze(0).to(device)

            with torch.no_grad():
                prediction = model(image)

            steering = float(prediction[0, 0].cpu())
            throttle = float(prediction[0, 1].cpu())

            # Keep predictions inside simulator limits
            steering = np.clip(steering, -1.0, 1.0)
            throttle = np.clip(throttle, 0.0, 1.0)
            action = [steering, throttle, 0.0]
            obs, reward, terminated, truncated, info = env.step(action)

            if step % 20 == 0:
                print(f"step={step:5d} | steering={steering:+.3f} | throttle={throttle:.3f}")

            if terminated or truncated:
                obs, info = env.reset()

    except KeyboardInterrupt:
        print("\nTest interrupted.")
    finally:
        env.close()
    print("Test finished.")
    exit(0)


# ============================================================
# LOAD DATA
# ============================================================

# Loads the dataset from the specified tub directories

tub_paths = [path.strip() for path in args.tubs.split(",") if path.strip()]
dataset = DonkeyBCDataset(tub_paths)


# ============================================================
# TRAIN / VALIDATION
# ============================================================

# Splits the dataset into training and validation data

train_indices, val_indices = [], []

for start, end in dataset.tub_boundaries:
    n = end - start
    n_val = int(n * 0.10)
    split_point = end - n_val

    train_indices.extend(range(start, split_point))
    val_indices.extend(range(split_point, end))

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

print(f"Training samples:   {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")


train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=(device.type == "cuda")
)

val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=(device.type == "cuda")
)


# ============================================================
# OPTIMIZER / LOSS
# ============================================================
#??????????????
# Defines the loss function and optimizer for training the model.

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


# ============================================================
# TRAINING
# ============================================================

# Trains the BC model using the training dataset.
# Evaluates on the validation dataset, and saves the best model based on validation loss.
# Implements early stopping based on the specified patience.

output_path = Path(args.output)
output_path.parent.mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(log_dir=f"runs/{output_path.stem}")
best_val_loss = float("inf")
epochs_without_improvement = 0
best_state = None


print()
print("===================================")
print("BC TRAINING")
print("===================================")
print("Epochs:", args.epochs)
print("Batch size:", args.batch_size)
print("Learning rate:", args.lr)
print("Early stopping patience:", args.patience)
print("Output:", output_path)
print()


for epoch in range(args.epochs):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()
    train_loss = 0.0
    progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad()
        predictions = model(images)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        progress.set_postfix(loss=f"{loss.item():.5f}")

    train_loss /= len(train_dataset)

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()
    val_loss = 0.0

    with torch.no_grad():

        for images, targets in val_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            predictions = model(images)
            loss = criterion(predictions, targets)

            val_loss += loss.item() * images.size(0)

    val_loss /= len(val_dataset)

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print(f"Epoch {epoch + 1:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)


    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        best_state = copy.deepcopy(model.state_dict())
        torch.save(best_state, output_path)
        print(f"!! New best model saved ({best_val_loss:.6f}) !!")

    else:
        epochs_without_improvement += 1
        print(f"  No improvement ({epochs_without_improvement}/{args.patience})")

    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if epochs_without_improvement >= args.patience:
        print()
        print("Early stopping triggered.")
        break


# ============================================================
# RESTORE BEST MODEL
# ============================================================

# Restores the best model state after training is complete.

if best_state is not None:
    model.load_state_dict(best_state)

writer.close()

print()
print("===================================")
print("TRAINING COMPLETE")
print("===================================")
print("Best validation loss:", best_val_loss)
print("Model saved to:", output_path)

model.eval()
