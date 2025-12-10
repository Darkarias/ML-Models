# GR_PINN

A Physics-Informed Neural Network (PINN) for solving geodesic motion in the Schwarzschild spacetime using natural units.

## Focus Keyword: GR PINN

This work introduces StarNet, a Physics-Informed Neural Network (PINN) designed to simulate geodesic motion in the curved
spacetime of a Schwarzschild black hole without relying on traditional numerical integration. By directly embedding the geodesic
equation and Christoffel symbols into the loss function, the network learns the geometry of spacetime as a continuous, differentiable
function of proper time. The results demonstrate that StarNet successfully captures non-Newtonian relativistic phenomena,
including perihelion precession and horizon crossing, while maintaining orbital stability over long integration times. This
approach offers a novel, data-free method for modeling general relativity, with potential applications for inverse problems in
astrophysics.

---

## Features
- PINN architecture with positional encoding
- Schwarzschild metric implementation
- Geodesic ODE residuals
- Initial condition constraints
- Radial trajectory logging
- CPU and GPU support
- Modular project layout

---

## Project Structure
```
GR_PINN/
│
├── train.py               # Training loop and logging
├── pinn.py                # PINN model, metric, residuals, IC loss
├── outputs/               # Generated trajectory plots
└── README.md              # Project documentation
```

---

## Installation
```bash
pip install -r requirements.txt
```

Minimum packages:
```
torch
numpy
matplotlib
```

---

## Training
Run training from the project root:
```bash
python train.py --steps 10000 --lr 1e-4 --M 1.0
```

---

## Logging
Every N steps the trainer logs:
- total loss
- residual terms
- initial condition error
- debug stats (r min/max, theta min/max)

Trajectory plots are saved automatically as versioned files:
```
outputs/radial_traj_v1.png
outputs/radial_traj_v2.png
```

---

## Code Highlights
### Metric
Implements Schwarzschild metric in natural units:
```
f(r) = 1 - 2M/r
```

### Residuals
Geodesic equations derived from:
```
d^2 x^μ / dτ^2 + Γ^μ_{αβ} (dx^α/dτ)(dx^β/dτ) = 0
```

### Loss Function
- Metric residuals
- Velocity normalization
- Initial condition matching

---

## Outputs
Training produces:
- Loss logs
- Versioned radial trajectory plots
- Optional checkpoint saves

---

## Future Work
- Add Kerr extension
- Add adaptive sampling
- Add tensorboard-style dashboard
- Add coordinate transforms

---

## License
MIT