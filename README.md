# dune-tension

Wire-tension measurement tooling for DUNE APA work, including:
- A Tk GUI for guided measurements (`dune_tension`)
- Spectrum/pitch analysis CLIs (`spectrum_analysis`)
- PLC/servo/valve control integration
- Logging, summaries, plots, and M2M upload utilities

## Repository Layout
- `src/dune_tension/`: installable GUI/runtime package
- `src/dune_tension/ukapa7_comparison/`: UKAPA7-specific comparison and report-generation helpers
- `src/dune_tension/hardware/`: hardware-facing helpers such as the valve trigger
- `src/spectrum_analysis/`: installable spectrum/pitch analysis package
- `tools/`: ad hoc developer utilities not shipped as packages
- `experiments/`: exploratory scripts and datasets kept out of the install path

## Requirements
- Python `>=3.12`
- macOS/Linux with audio input support
- Optional hardware:
  - PLC reachable by `src/dune_tension/tension_server.py`
  - Pololu Micro Maestro servo controller
  - Supported valve trigger device

Project dependencies are declared in [pyproject.toml](pyproject.toml).

## Install

### Option 1: uv (recommended)
```bash
uv sync
```

### Option 2: venv + pip
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Entry Points
After install, the following console commands are available:

- `dune-tension-gui`
- `dune-tension-periodic-plots`
- `dune-spectrum-compare`
- `dune-spectrum-scroller`

## Quick Start

### 1. Start the PLC tension server (machine connected to PLC)
```bash
python3 -m dune_tension.tension_server
```

Edit `PLC_IP_ADDRESS`/`SERVER_PORT` in [src/dune_tension/tension_server.py](src/dune_tension/tension_server.py) as needed.

### 2. Launch the GUI
```bash
dune-tension-gui
```

The GUI lets you configure APA/layer/side, collect measurements, clear/re-measure ranges, and refresh summary outputs.

### 3. Generate summary outputs periodically (optional)
```bash
dune-tension-periodic-plots USAPA5 X --interval 900
```

## Hardware and Network Notes
- Ensure the PLC server host is reachable from the GUI machine.
- Configure audio input device (historically "USB PnP" style device names).
- Servo controller reference: <https://www.pololu.com/product/1350/resources>

## Spoof / Dry-Run Modes
Set environment variables before launching the GUI:

- `SPOOF_AUDIO=1` : use spoofed audio capture
- `SPOOF_PLC=1` : bypass real PLC movement
- `SPOOF_SERVO=1` : use dummy servo controller
- `SPOOF_VALVE=1` : disable real valve control

Example:
```bash
SPOOF_AUDIO=1 SPOOF_PLC=1 SPOOF_SERVO=1 SPOOF_VALVE=1 dune-tension-gui
```

## Data and Output Paths
Default runtime paths include:
- Measurement DB: `data/tension_data/tension_data.db`
- Summaries: `data/tension_summaries/`
- Plots: `data/tension_plots/`
- Missing/bad wires: `data/badwires/`

## Spectrum / Pitch Tools

### Compare pitch workflows
```bash
dune-spectrum-compare --config src/spectrum_analysis/pitch_compare_config.json
```

### Scrolling spectrogram viewer
```bash
dune-spectrum-scroller --demo
```

## M2M Upload Utilities
M2M helpers live in `src/dune_tension/m2m/`.

Example uploader script:
- [src/dune_tension/uploadTensions.py](src/dune_tension/uploadTensions.py)

## Development

### Compile check
```bash
python3 -m compileall src
```

### Tests
If not already installed:
```bash
pip install pytest
```

Then run:
```bash
pytest
```

## Troubleshooting
- If audio/GUI imports fail, verify system audio libs and `sounddevice` install.
- If PLC movement fails, confirm `TENSION_SERVER_URL` in [src/dune_tension/plc_io.py](src/dune_tension/plc_io.py).
- If `uv` is not found after install, add `~/.local/bin` to your shell `PATH`.
