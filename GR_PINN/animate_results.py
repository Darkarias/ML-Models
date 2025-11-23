import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
import argparse
from pathlib import Path

# Import your model
from pinn import StarNet

def get_trajectory(model, M, r_start, vr_start, vphi_start, tau_range=(-10, 10), steps=2000, device='cpu'):
    """Generates the trajectory data from the model."""
    model.eval()
    t_vals = np.linspace(tau_range[0], tau_range[1], steps)
    tau_tensor = torch.tensor(t_vals, dtype=torch.float32).view(-1, 1).to(device)
    
    rs = 2.0 * M
    phi_start = 0.0
    theta_start = np.pi / 2.0 
    
    N = len(t_vals)
    p0 = torch.tensor([[0.0, r_start, theta_start, phi_start]], dtype=torch.float32, device=device).repeat(N, 1)
    
    f = 1.0 - rs / r_start
    spatial_term = (1.0/f)*(vr_start**2) + (r_start**2)*(vphi_start**2)
    vt_start = np.sqrt((1.0 + spatial_term) / f)
    
    v0 = torch.tensor([[vt_start, vr_start, 0.0, vphi_start]], dtype=torch.float32, device=device).repeat(N, 1)
    
    inputs = torch.cat([p0, v0, tau_tensor], dim=1)
    
    with torch.no_grad():
        coords = model(inputs, M)
        
    coords = coords.cpu().numpy()
    
    return coords[:, 1], coords[:, 2], coords[:, 3], t_vals

def create_2d_animation(r, theta, phi, M, save_path, downsample=1):
    print(f"Generating 2D Animation to {save_path}...")
    
    # Downsample data for smoother/smaller GIF
    r = r[::downsample]
    phi = phi[::downsample]
    
    rs = 2.0 * M
    isco = 6.0 * M
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Static Elements
    horizon = plt.Circle((0, 0), rs, color='black', label='Event Horizon', zorder=10)
    ax.add_artist(horizon)
    isco_circle = plt.Circle((0, 0), isco, color='red', fill=False, linestyle='--')
    ax.add_artist(isco_circle)
    
    # Dynamic Elements
    trail, = ax.plot([], [], color='orange', linewidth=1.5, alpha=0.8)
    particle, = ax.plot([], [], 'go', markersize=8, label='Particle')
    
    # Axis Limits
    limit = max(np.max(np.abs(x)), np.max(np.abs(y)), 10*M) * 1.1
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title("2D Orbit Animation")

    def update(frame):
        # Update trail up to current frame
        trail.set_data(x[:frame], y[:frame])
        # Update particle position
        particle.set_data([x[frame]], [y[frame]])
        return trail, particle

    # Create Animation
    # interval=20 means 20ms per frame (50 fps)
    anim = FuncAnimation(fig, update, frames=len(x), interval=20, blit=True)
    anim.save(save_path, writer=PillowWriter(fps=30))
    plt.close()

def create_3d_animation(r, theta, phi, M, save_path, downsample=1):
    print(f"Generating 3D Animation to {save_path}...")
    
    r = r[::downsample]
    theta = theta[::downsample]
    phi = phi[::downsample]
    
    rs = 2.0 * M
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw Sphere (Static)
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:20j]
    sx = rs * np.cos(u) * np.sin(v)
    sy = rs * np.sin(u) * np.sin(v)
    sz = rs * np.cos(v)
    ax.plot_surface(sx, sy, sz, color='black', alpha=0.8)
    
    # Dynamic Elements
    trail, = ax.plot([], [], [], color='cyan', linewidth=1.5)
    particle, = ax.plot([], [], [], 'go', markersize=8)
    
    limit = max(np.max(np.abs(x)), np.max(np.abs(y)), 10*M) * 1.1
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit/2, limit/2)
    
    def update(frame):
        trail.set_data(x[:frame], y[:frame])
        trail.set_3d_properties(z[:frame])
        
        particle.set_data([x[frame]], [y[frame]])
        particle.set_3d_properties([z[frame]])
        return trail, particle

    anim = FuncAnimation(fig, update, frames=len(x), interval=20, blit=False)
    anim.save(save_path, writer=PillowWriter(fps=30))
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="training_outputs/latest_checkpoint.pth")
    parser.add_argument("--tau-range", type=float, default=20.0)
    parser.add_argument("--mass", type=float, default=0.5)
    
    parser.add_argument("--radius", type=float, default=10.0)
    parser.add_argument("--vr", type=float, default=0.0)
    parser.add_argument("--vphi", type=float, default=0.09)
    
    # Animation specific args
    parser.add_argument("--steps", type=int, default=5000, help="Total physics steps to calculate")
    parser.add_argument("--frames", type=int, default=300, help="Number of frames in the GIF (downsamples automatically)")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StarNet().to(device)
    
    if Path(args.checkpoint).exists():
        print(f"Loading weights from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    r_start_val = args.radius * args.mass 
    
    print(f"Calculating physics ({args.steps} steps)...")
    r, th, phi, _ = get_trajectory(
        model, args.mass, 
        r_start=r_start_val, vr_start=args.vr, vphi_start=args.vphi,
        tau_range=(-args.tau_range, args.tau_range), 
        steps=args.steps, device=device
    )
    
    # Calculate downsample rate to hit target frame count
    downsample = max(1, args.steps // args.frames)
    
    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)
    
    create_2d_animation(r, th, phi, args.mass, output_dir / "orbit_2d.gif", downsample)
    create_3d_animation(r, th, phi, args.mass, output_dir / "orbit_3d.gif", downsample)
    print("Done!")

if __name__ == "__main__":
    main()