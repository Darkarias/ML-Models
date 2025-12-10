import argparse
from pathlib import Path
import random
import numpy as np
import torch
import os

from GR_PINN.pinn import (StarNet, calculate_total_loss, sample_collocation)

try:
    from GR_PINN.visualization import log_radial_trajectory
except ModuleNotFoundError:
    log_radial_trajectory = None

def set_seed(seed):
    """
    Set random seeds for reproducibility across all libraries.
    Ensures training runs are deterministic and can be replicated.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    # Disable non-deterministic operations for full reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(
    steps=1500,
    batch_size=64,
    lr=1e-5,
    lambda_phys=1.0,
    lambda_ic=1.0,
    M=0.5,
    tau_min=-1.0,
    tau_max=1.0,
    log_interval=50,
    eval_interval=200,
    traj_samples=200,
    output_dir=Path("training_outputs"),
    device=None,
    random_seed=1234,
    resume_path=None,
    save_interval=1000,
):

    """
    Main training loop for the Physics-Informed Neural Network.
    
    Args:
        steps: Total number of training iterations
        batch_size: Number of collocation points per batch
        lr: Learning rate for Adam optimizer
        lambda_phys: Weight for physics loss (geodesic equation)
        lambda_ic: Weight for initial condition loss
        M: Black hole mass parameter
        tau_min/tau_max: Range of proper time for sampling
        log_interval: Steps between loss logging
        eval_interval: Steps between trajectory visualization
        traj_samples: Number of points for trajectory plots
        output_dir: Directory for saving checkpoints and plots
        device: Device to train on (cuda/cpu)
        random_seed: Seed for reproducibility
        resume_path: Path to checkpoint file to resume training
        save_interval: Steps between saving checkpoints
    """

    set_seed(random_seed)
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    model = StarNet().to(device)
    # Adam optimizer for efficient gradient-based learning
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Warn if visualization module is missing
    if log_radial_trajectory is None:
        print("Visualization disabled (matplotlib module not available).")
    
    # Initialize training step counter
    start_step = 1
    # Resume from checkpoint if provided
    if resume_path and os.path.exists(resume_path):
        print(f"Loading checkpoint from {resume_path}...")
        checkpoint = torch.load(resume_path, map_location=device)
        # Restore model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        # Restore optimizer state (important for momentum-based optimizers)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Resume from the next step after the checkpoint
        if 'step' in checkpoint:
            start_step = checkpoint['step'] + 1
        print(f"Resumed from step {start_step}")
    else:
        if resume_path:
            print(f"Warning: Resume path {resume_path} provided but file not found. Starting from scratch.")

    # Define the proper time sampling range
    tau_range = (tau_min, tau_max)

    print(f"Starting training on {device}...")
    
    # Main training loop
    for step in range(start_step, steps + 1):
        # Sample random collocation points: initial conditions + random times
        # Physics loss is evaluated at these points
        physics_data, ic_data = sample_collocation(batch_size, M=M, tau_range=tau_range, device=str(device))

        # Compute combined loss: physics (geodesic) + initial conditions
        loss = calculate_total_loss(model, physics_data, ic_data, M, lambda_phys=lambda_phys, lambda_ic=lambda_ic)

        # Standard PyTorch training step
        optimizer.zero_grad() # Clear previous gradients
        loss.backward() # Compute gradients via backpropagation
        # Clip gradients to prevent explosion in unstable physics regions
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step() # Update model parameters


        # Log training progress periodically
        if step % log_interval == 0:
            print(f"[step {step:05d}] total_loss={loss.item():.6f}")

        # Save checkpoint periodically to enable resuming training
        if step % save_interval == 0 or step == steps:
            save_path = output_dir / "latest_checkpoint.pth"
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, save_path)
            print(f"Saved checkpoint to {save_path}")


        # Generate and save trajectory visualization periodically
        if step % eval_interval == 0 and log_radial_trajectory is not None:
            plot_path = output_dir / f"radial_traj_step{step:05d}.png"
            log_radial_trajectory(
                model,
                M=M,
                tau_range=tau_range,
                traj_samples=traj_samples,
                save_path=plot_path,
                device=device,
            )

def parse_args():
    """
    Parse command-line arguments for training configuration.
    Allows flexible hyperparameter tuning without code modification.
    """
    parser = argparse.ArgumentParser(description="Train Schwarzschild PINN.")
    
    # Training duration and batch configuration
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    
    # Optimizer hyperparameters
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lambda-phys", type=float, default=10.0)
    parser.add_argument("--lambda-ic", type=float, default=10.0)
    parser.add_argument("--mass", type=float, default=0.5)
    
    # Physics domain parameters (Default to small domain for stability)
    parser.add_argument("--tau-min", type=float, default=-2.0)
    parser.add_argument("--tau-max", type=float, default=2.0)
    
    # Logging and checkpointing
    parser.add_argument("--log-interval", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--save-interval", type=int, default=1000, help="Steps between checkpoint saves")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    
    # Visualization and output
    parser.add_argument("--traj-samples", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=Path("training_outputs"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()
    # Set device from arguments if provided
    device = torch.device(args.device) if args.device else None

    # Launch training with all specified parameters
    train(
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_phys=args.lambda_phys,
        lambda_ic=args.lambda_ic,
        M=args.mass,
        tau_min=args.tau_min,
        tau_max=args.tau_max,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        traj_samples=args.traj_samples,
        output_dir=args.output_dir,
        device=device,
        random_seed=args.seed,
        resume_path=args.resume,
        save_interval=args.save_interval,
    )