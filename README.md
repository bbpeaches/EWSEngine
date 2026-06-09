# EWSEngine

Version: 0.2.3

EWSEngine is a reorganized electromagnetic wave simulation project. It has a typed backend physics layer, a local HTTP API, and a desktop launcher built on Tkinter and Matplotlib.

## Structure

- `backend/`: reusable simulation engines, dataclasses, and geometry helpers
- `api/`: local HTTP API exposing the same engines used by the desktop frontend
- `frontend/`: launcher, scene base class, plotting helpers, and migrated scenes
- `core/`: exceptions, registry, and shared type definitions
- `modules/`: compatibility exports for older scene import paths
- `run.py`: startup script for desktop and API modes

## Dependencies

Install pip dependencies:

```bash
python -m pip install -r requirements.txt
```

Tkinter is not distributed as a normal pip package. With Homebrew Python 3.13, install the matching Tk bridge separately:

```bash
brew install python-tk@3.13
```

`brew info python-tk@3.13` reports it as the Python interface to Tcl/Tk, depending on `python@3.13` and `tcl-tk`. In this environment it is already installed and linked.

## Start

Run desktop + API:

```bash
python run.py
```

Run API only:

```bash
python run.py --api-only
```

Run desktop only:

```bash
python run.py --desktop-only
```

## Desktop Modules

The launcher currently exposes six simulations:

- `optics`: interface optics, coplanarity, total internal reflection, Brewster angle
- `polarization`: phase synthesis, circular-basis synthesis, antenna polarization match
- `transmission`: standing-wave envelope and traveling/standing decomposition
- `wave`: material wave, lossy medium, constant-phase plane visualization
- `tem`: TEM uniform plane-wave propagation with direction and polarity controls
- `speed`: dispersion and apparent phase-speed geometry

The desktop scenes share a common control framework with:

- pause / resume
- zoom in / out / reset
- reset view
- per-scene presets
- per-scene mode switching

## API

- `GET /health`
- `GET /modules`
- `GET /modules/{key}`
- `POST /simulate/{key}`

Example:

```bash
curl http://127.0.0.1:8765/modules
curl -X POST http://127.0.0.1:8765/simulate/optics \
  -H 'Content-Type: application/json' \
  -d '{"n1": 1.0, "n2": 1.5, "theta_deg": 35.0, "phi_deg": 20.0}'
```

## Checks

```bash
python -m compileall .
python -m pytest
```

Current automated verification covers:

- physics invariants for Fresnel optics, total internal reflection, polarization classification, and transmission envelopes
- service-level simulation dispatch for all registered modules
- HTTP API health, module listing, and optics simulation endpoint
- noninteractive frontend smoke tests for all desktop scene classes
