from pathlib import Path
import torch

def next_versioned_filename(base_path: Path) -> Path:
    stem = base_path.stem
    suffix = base_path.suffix
    parent = base_path.parent

    version = 1
    while True:
        candidate = parent / f"{stem}_v{version}{suffix}"
        if not candidate.exists():
            return candidate
        version += 1

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ModuleNotFoundError:
    HAS_MATPLOTLIB = False
    plt = None

from pinn import sample_initial_conditions


@torch.no_grad()
def log_radial_trajectory(model, M, tau_range, traj_samples, save_path: Path, device):
    """Roll out one geodesic and plot r(τ) to check horizon crossings."""
    if not HAS_MATPLOTLIB:
        print("Skipping trajectory plot: matplotlib is not installed.")
        return

    # Create versioned filename so plots never overwrite
    save_path = next_versioned_filename(save_path)

    rs = 2.0 * M
    p0, v0 = sample_initial_conditions(batch_size=1, rs=rs, device=device)
    tau = torch.linspace(tau_range[0], tau_range[1], traj_samples, device=device).unsqueeze(1)

    init_p = p0.repeat(traj_samples, 1)
    init_v = v0.repeat(traj_samples, 1)
    coords = model(torch.cat([init_p, init_v, tau], dim=1))
    r_vals = coords[:, 1].cpu()
    tau_vals = tau[:, 0].cpu()

    plt.figure(figsize=(6, 4))
    plt.axhline(rs, color="red", linestyle="--", label="Schwarzschild radius")
    plt.plot(tau_vals, r_vals, label="Predicted r(τ)")
    plt.xlabel("Proper time τ")
    plt.ylabel("Radial coordinate r")
    plt.title("Radial trajectory sanity check")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()