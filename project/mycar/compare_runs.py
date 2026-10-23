"""Plot TensorBoard scalars from several runs on one figure (e.g. PPO vs SAC).

    python compare_runs.py                              # all runs under models/ppo_models and models/sac_models
    python compare_runs.py models/ppo_models/a models/sac_models/b
    python compare_runs.py --out compare.png --show
"""
import argparse
import os

import matplotlib

TAGS = ["rollout/ep_rew_mean", "rollout/ep_len_mean", "rollout/avg_speed"]
DEFAULT_ROOTS = ["models/ppo_models", "models/sac_models"]

parser = argparse.ArgumentParser(description="Compare TensorBoard runs (reward, episode length, speed vs timesteps)")
parser.add_argument("runs", nargs="*", help="Run dirs (containing tb_logs/) or any dir with event files. Default: everything under models/ppo_models and models/sac_models.")
parser.add_argument("--out", type=str, default="compare_runs.png", help="Output PNG path.")
parser.add_argument("--show", action="store_true", help="Open the plot window as well.")
parser.add_argument("--smooth", type=float, default=0.0, help="EMA smoothing 0-1 (0 = raw).")
args = parser.parse_args()

if not args.show:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def find_event_dirs(root):
    """Dirs under root that directly contain tfevents files."""
    return sorted(d for d, _, files in os.walk(root) if any(f.startswith("events.out.tfevents") for f in files))


def run_name(event_dir, root):
    """Label like 'ppo_session3' (the run dir), skipping tb_logs/<name>_0 plumbing."""
    parts = os.path.relpath(event_dir, root).split(os.sep)
    parts = [p for p in parts if p != "tb_logs"]
    if len(parts) > 1 and parts[-1].rsplit("_", 1)[0] == parts[0]:
        parts = parts[:-1]     # tb_log_name_0 duplicates the run dir name
    return "/".join(parts) if parts[0] != "." else os.path.basename(os.path.abspath(root))


# Collect (label, event_dir) pairs
runs = []
if args.runs:
    for r in args.runs:
        dirs = find_event_dirs(r)
        if not dirs:
            print(f"No event files under {r}, skipping")
        base = os.path.dirname(os.path.normpath(r)) or "."
        runs += [(run_name(d, base), d) for d in dirs]
else:
    for root in DEFAULT_ROOTS:
        runs += [(run_name(d, root), d) for d in find_event_dirs(root)]

# Same run name under ppo_models and sac_models: disambiguate with the parent dir
labels = [l for l, _ in runs]
if len(set(labels)) < len(labels):
    runs = [(os.path.basename(d.split("/tb_logs")[0].rstrip("/").rsplit("/", 1)[0]) + "/" + l if labels.count(l) > 1 else l, d) for l, d in runs]

if not runs:
    raise SystemExit("No TensorBoard event files found. Run from project/mycar or pass run dirs.")

# Load scalars: label -> tag -> (steps, values). Several event dirs per run (resumed) are merged by step.
data = {}
for label, d in runs:
    acc = EventAccumulator(d, size_guidance={"scalars": 0})
    acc.Reload()
    available = set(acc.Tags()["scalars"])
    for tag in TAGS:
        if tag in available:
            ev = acc.Scalars(tag)
            data.setdefault(label, {}).setdefault(tag, []).extend((e.step, e.value) for e in ev)
    if not available & set(TAGS):
        print(f"{label}: none of {TAGS} logged (has: {sorted(available)})")

for label in data:
    for tag in data[label]:
        data[label][tag].sort()


def ema(vals, a):
    out, last = [], None
    for v in vals:
        last = v if last is None else a * last + (1 - a) * v
        out.append(last)
    return out


present = [t for t in TAGS if any(t in data[l] for l in data)]
missing = [t for t in TAGS if t not in present]
if missing:
    print("Not found in any run:", ", ".join(missing))
if not present:
    raise SystemExit("None of the expected tags found in the given runs.")

fig, axes = plt.subplots(len(present), 1, figsize=(9, 3.2 * len(present)), sharex=True, squeeze=False)
for ax, tag in zip(axes[:, 0], present):
    for label in sorted(data):
        if tag not in data[label]:
            continue
        steps, vals = zip(*data[label][tag])
        if args.smooth > 0:
            vals = ema(vals, args.smooth)
        ax.plot(steps, vals, label=label, linewidth=1.5)
    ax.set_title(tag)
    ax.set_ylabel(tag.split("/")[1])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
axes[-1, 0].set_xlabel("timesteps")
fig.tight_layout()
fig.savefig(args.out, dpi=120)
print(f"Saved {args.out}  ({len(data)} runs, tags: {', '.join(present)})")
if args.show:
    plt.show()
