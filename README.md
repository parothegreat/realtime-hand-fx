# Realtime Hand FX

I built this because I was bored.

The idea came from those interactive visual effects people usually make in
TouchDesigner. I was too lazy to build it there, so I made my own version in
Python instead lol.

## Windows

Use Windows 10 or 11 with 64-bit Python 3.12. Run these commands in PowerShell:

```powershell
git clone https://github.com/parothegreat/realtime-hand-fx.git
cd realtime-hand-fx
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

## Linux

This is the setup for Fedora:

```bash
git clone https://github.com/parothegreat/realtime-hand-fx.git
cd realtime-hand-fx
sudo dnf install python3-gobject gstreamer1-plugins-base \
  gstreamer1-plugins-base-tools gstreamer1-plugins-good \
  gstreamer1-plugins-bad-free
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

To switch cameras, change `CAM_INDEX` in `main.py`. The program will figure out
the resolution, frame rate, camera format, and GPU backend on its own.

If you want to check that the tracking and renderer work before opening the
camera:

```bash
python main.py --self-check
python main.py --gpu-check
```
