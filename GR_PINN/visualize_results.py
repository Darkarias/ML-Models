import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
from pathlib import Path

# Import your model architecture
from pinn import StarNet

def get_trajectory(model, M, r_start, vr_start, vphi_start, tau_range=(-10, 10), steps=2000, device='cpu'):
    """
    Runs the model over a range of Proper Time (tau) to generate the full orbit.
    """
    model.eval()
    t_vals = np.linspace(tau_range[0], tau_range[1], steps)
    # Force Float32
    tau_tensor = torch.tensor(t_vals, dtype=torch.float32).view(-1, 1).to(device)
    
    rs = 2.0 * M
    
    # Fixed Initial Conditions based on arguments
    phi_start = 0.0
    theta_start = np.pi / 2.0 # Equatorial plane
    
    # Construct input batch (N, 9)
    N = len(t_vals)
    
    p0 = torch.tensor([[0.0, r_start, theta_start, phi_start]], 
                      dtype=torch.float32, device=device).repeat(N, 1)
    
    # Calculate vt for mass shell condition
    f = 1.0 - rs / r_start
    spatial_term = (1.0/f)*(vr_start**2) + (r_start**2)*(vphi_start**2)
    vt_start = np.sqrt((1.0 + spatial_term) / f)
    
    v0 = torch.tensor([[vt_start, vr_start, 0.0, vphi_start]], 
                      dtype=torch.float32, device=device).repeat(N, 1)
    
    # Combine into input vector
    inputs = torch.cat([p0, v0, tau_tensor], dim=1)
    
    with torch.no_grad():
        coords = model(inputs, M)
        
    coords = coords.cpu().numpy()
    r = coords[:, 1]
    theta = coords[:, 2]
    phi = coords[:, 3]
    
    return r, theta, phi, t_vals

def plot_2d_orbit(r, theta, phi, M, save_path=None):
    rs = 2.0 * M
    isco = 6.0 * M
    
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    
    # Draw Event Horizon & ISCO
    horizon = plt.Circle((0, 0), rs, color='black', label='Event Horizon ($2M$)', zorder=10)
    ax.add_artist(horizon)
    isco_circle = plt.Circle((0, 0), isco, color='red', fill=False, linestyle='--', label='ISCO ($6M$)')
    ax.add_artist(isco_circle)
    
    # Plot Trajectory
    ax.scatter(x, y, c=np.linspace(0, 1, len(x)), cmap='plasma', s=5, label='Trajectory')
    ax.plot(x[0], y[0], 'go', label='Start', markersize=10)
    ax.plot(x[-1], y[-1], 'rx', label='End', markersize=10)
    
    ax.set_aspect('equal')
    limit = max(np.max(np.abs(x)), np.max(np.abs(y)), 10*M) * 1.1
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    plt.xlabel("X ($M$)")
    plt.ylabel("Y ($M$)")
    plt.title(f"Orbit (M={M})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved 2D plot to {save_path}")
    plt.close()

def plot_3d_orbit(r, theta, phi, M, save_path=None):
    rs = 2.0 * M
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw Sphere
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:20j]
    sx = rs * np.cos(u) * np.sin(v)
    sy = rs * np.sin(u) * np.sin(v)
    sz = rs * np.cos(v)
    ax.plot_surface(sx, sy, sz, color='black', alpha=0.8)
    
    ax.plot(x, y, z, color='cyan', linewidth=2)
    ax.scatter(x[0], y[0], z[0], color='green', s=50)
    ax.scatter(x[-1], y[-1], z[-1], color='red', s=50, marker='x')
    
    limit = max(np.max(np.abs(x)), np.max(np.abs(y))) * 1.1
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit/2, limit/2)
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved 3D plot to {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="training_outputs/latest_checkpoint.pth")
    parser.add_argument("--tau-range", type=float, default=100.0)
    parser.add_argument("--mass", type=float, default=0.5)
    
    # Physics Control
    parser.add_argument("--radius", type=float, default=6.0, help="Initial Radius (M units)")
    parser.add_argument("--vr", type=float, default=0.0, help="Initial Radial Velocity")
    parser.add_argument("--vphi", type=float, default=0.1, help="Initial Angular Velocity")
    
    # Resolution Control
    parser.add_argument("--steps", type=int, default=2000, help="Number of points to plot")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StarNet().to(device)
    
    if Path(args.checkpoint).exists():
        print(f"Loading weights from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    r_start_val = args.radius * args.mass 
    
    print(f"Simulating: Start r={r_start_val:.2f}, vr={args.vr}, vphi={args.vphi}")
    
    r, th, phi, t = get_trajectory(
        model, args.mass, 
        r_start=r_start_val, 
        vr_start=args.vr, 
        vphi_start=args.vphi,
        tau_range=(-args.tau_range, args.tau_range), 
        steps=args.steps,  # <--- PASSING STEPS HERE
        device=device
    )
    
    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)
    
    plot_2d_orbit(r, th, phi, args.mass, save_path=output_dir / "orbit_2d.png")
    plot_3d_orbit(r, th, phi, args.mass, save_path=output_dir / "orbit_3d.png")

if __name__ == "__main__":
    main()