import argparse
from pathlib import Path
import random
import numpy as np
import torch
import os

from GR_PINN.PINN.pinn import (
    StarNet,
    calculate_total_loss,
    sample_collocation,
)

try:
    from GR_PINN.PINN.visualization import log_radial_trajectory
except ModuleNotFoundError:
    log_radial_trajectory = None

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
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

    set_seed(random_seed)
    device = device or (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    model = StarNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    output_dir.mkdir(parents=True, exist_ok=True)

    if log_radial_trajectory is None:
        print("Visualization disabled (matplotlib module not available).")

    # --- RESUME LOGIC ---
    start_step = 1
    if resume_path and os.path.exists(resume_path):
        print(f"Loading checkpoint from {resume_path}...")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # We start from the step AFTER the saved step
        if 'step' in checkpoint:
            start_step = checkpoint['step'] + 1
        print(f"Resumed from step {start_step}")
    else:
        if resume_path:
            print(f"Warning: Resume path {resume_path} provided but file not found. Starting from scratch.")
    # --------------------

    tau_range = (tau_min, tau_max)

    print(f"Starting training on {device}...")
    
    for step in range(start_step, steps + 1):
        physics_data, ic_data = sample_collocation(
            batch_size,
            M=M,
            tau_range=tau_range,
            device=device,
        )

        loss = calculate_total_loss(
            model,
            physics_data,
            ic_data,
            M,
            lambda_phys=lambda_phys,
            lambda_ic=lambda_ic,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # ---------- DEBUG BLOCK ----------
        if step % log_interval == 0:
            print(f"[step {step:05d}] total_loss={loss.item():.6f}")
        # --------------------------------

        # --- SAVE CHECKPOINT ---
        if step % save_interval == 0 or step == steps:
            save_path = output_dir / "latest_checkpoint.pth"
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, save_path)
            print(f"Saved checkpoint to {save_path}")
        # -----------------------

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
    parser = argparse.ArgumentParser(description="Train Schwarzschild PINN.")
    
    # Training Length
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    
    # Hyperparameters
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lambda-phys", type=float, default=10.0)
    parser.add_argument("--lambda-ic", type=float, default=10.0)
    parser.add_argument("--mass", type=float, default=0.5)
    
    # Physics Domain (Default to small domain for stability)
    parser.add_argument("--tau-min", type=float, default=-2.0)
    parser.add_argument("--tau-max", type=float, default=2.0)
    
    # Logging & Saving
    parser.add_argument("--log-interval", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--save-interval", type=int, default=1000, help="Steps between checkpoint saves")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    
    parser.add_argument("--traj-samples", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=Path("training_outputs"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device) if args.device else None

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