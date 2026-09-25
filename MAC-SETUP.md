# RF Lab Session 1 on macOS (Apple Silicon)

Translation of Antton's Session1.pptx, which assumes Windows lab PCs.

## What does NOT apply on Mac

| Slide says | On your Mac |
|---|---|
| Hyper-V | Ignore. Docker Desktop runs its own VM. |
| WSL2 box / "press any key to install" | Does not exist. No prompt will appear. |
| AMD64 downloads | Use arm64 / Apple Silicon builds. |
| Edge browser | Use Chrome for the gamepad page. Safari's gamepad support is unreliable. |
| ctrl+shift+p in VS Code | cmd+shift+p |
| ctrl + right click a link | cmd + click, or just paste the URL |

## Mac-only fixes (already applied)

1. Removed `"credsStore": "desktop"` from `~/.docker/config.json`.
   Without this every image pull dies with
   `docker-credential-desktop: executable file not found`.
   Backup: `~/.docker/config.json.bak-*`

2. Created `compose.mac.yml` — same as `compose.yml` minus the NVIDIA GPU
   reservation. Verified on this machine: the GPU block fails with
   `no known GPU vendor found`. There is no NVIDIA GPU on a Mac.
   Consequence: torch runs CPU-only. BC training on cropped greyscale
   images is small, so this is slow but fine.

## Steps

### 1. Build (you already have every file needed)
No clone required. `dockerfile`, `compose.yml`, `requirements.txt` and
`entrypoint.sh` are all here, and that is the whole image.

    cd ~/Desktop/RF
    docker compose -f compose.mac.yml up -d --build

First build is long: torch plus the CUDA wheels. Later starts:

    docker compose -f compose.mac.yml up -d

Always pass `-f compose.mac.yml`. The plain `compose.yml` demands an
NVIDIA GPU and cannot start on this machine.

### 2. Missing piece: carscript/
Slide 11 wants you to copy `carscript/` into `mycar/`. That folder is in
the DC.LAB GitLab repo, which you do not have access to. It holds the
course scripts: `preprocessing.py`, `train_bc.py`, `train.py`, and the
`myconfig.py` the slides tell you to edit.

Ask Antton for it in the lab. Everything up to step 4 works without it;
steps 5 and 8-11 do not.

### 3. SSH in
    ssh racer@localhost          # password: racer

If it refuses on a host key, clear the stale entry:

    ssh-keygen -R localhost

`racervenv` activates automatically. Check:

    python --version             # expect 3.12.3
    donkey --version             # expect 5.4.dev1

### 4. Create the car
    cd project
    donkey createcar --path mycar

### 5. Move the scripts
Copy everything from `carscript/` into `mycar/`.
`carscript` sits next to `project`.

### 6. VS Code
Install the Remote-SSH extension, then cmd+shift+p →
"Remote-SSH: Connect to Host" → `racer@localhost` → password `racer`.
Open the `project` folder.

### 7. Simulator - DONE, already downloaded
`~/Desktop/RF/DonkeySimMac/donkey_sim.app` - just double-click it.

Release v25.10.06 from tawnkramer/gym-donkeycar, the same project the
container installed. Universal binary with a native arm64 slice, so it
runs at full speed, no Rosetta. No quarantine flag, so Gatekeeper will
not block it.

The simulator runs on your Mac, NOT in the container. The container
connects to it over the network.

### 8. Task from slide 16
In `mycar/myconfig.py` set `controller_type` to `xbox`.

### 9. Collect data
In the container, inside `mycar`:

    python manage.py drive

Start the Warren scene in the simulator. Open http://localhost:8887
in Chrome, click Gamepad, connect the Xbox controller by cable, click
the page, drive. Recording runs while throttle > 0.

About 20 laps, then ctrl+c. Data lands in `mycar/data` as one tub.

### 10. Train
    python train_bc.py --tubs <tub path> --output models/bc_models/<name>.pth

### 11. Test
    python train_bc.py --model models/bc_models/<name>.pth --test

Run Warren, then Generated Road.

## Unverified

The full pip resolve against `requirements.txt` on arm64 was not run to
completion. Individual wheels all exist for arm64, including the NVIDIA
ones, but the two editable git installs (`donkeycar@ba25266`,
`gym-donkeycar@a1f4ca6`) were not resolution-tested. If the build dies,
that is the first place to look.

## Simulator config - the two lines that actually matter

Add these to `mycar/myconfig.py`. Without them the car will not move,
even though the camera feed works and the web page looks fine.

    DONKEY_GYM = True
    DONKEY_SIM_PATH = "remote"
    DONKEY_GYM_ENV_NAME = "donkey-warren-track-v0"
    SIM_HOST = "host.docker.internal"
    USE_JOYSTICK_AS_DEFAULT = False

`SIM_HOST` - the simulator runs on the Mac, the code runs in the
container. The default `127.0.0.1` means the container itself, so it
never finds the sim. `host.docker.internal` is Docker Desktop's name
for the Mac.

`USE_JOYSTICK_AS_DEFAULT = False` - config.py defaults this to True,
which makes donkeycar read a physical joystick at /dev/input/js0 on the
machine running donkeycar. That machine is the Linux container, and a
controller plugged into the Mac is not visible there. Result: the car
ignores the browser entirely while still streaming camera, which looks
like everything works. Setting this False routes control through the
web page at localhost:8887, which is what the slides intend.

Keep `CONTROLLER_TYPE = 'xbox'` COMMENTED OUT. Slide 16 says to set it,
but on Docker that adds the physical-joystick part and breaks web control.

## Do not run env.close() against a live sim

gym-donkeycar sends a quit message to the simulator on close, which
shuts the app down. If the sim vanishes mid-session, that is usually why.

## Restarting after a reboot

    cd ~/Desktop/RF
    docker compose -f compose.mac.yml up -d
    open DonkeySimMac/donkey_sim.app     # start the sim FIRST
    ssh racer@localhost                  # password: racer
    cd ~/project/mycar && python manage.py drive

Order matters: manage.py connects to the sim once at startup and does
not retry.
