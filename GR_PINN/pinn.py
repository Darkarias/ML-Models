import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class StarBlock(nn.Module):
    """Residual block with activation and dropout for the neural network."""
    def __init__(self, in_dim, out_dim, inter_dim_factor=1, drop_ratio=0.1, activation=F.softplus):
        super().__init__()
        inter_dim = inter_dim_factor * out_dim
        # Two layer transformation with intermediate dimension
        self.black_hole_in = nn.Linear(in_dim, inter_dim)
        self.black_hole_out = nn.Linear(inter_dim, out_dim)
        self.dropout = nn.Dropout(drop_ratio)
        self.activation = activation
    
    def forward(self, x):
        # Store input for residual connection.
        x0 = x
        # Apply transformation activation -> linear -> dropout.
        x = self.activation(self.black_hole_in(x))
        x = self.dropout(self.black_hole_out(x))
        # Add residual connection to help gradient flow.
        return x + x0

def positional_encoding(x, num_encoding=16):
    """
    Encode input coordinates using sine and cosine at different frequencies.
    Helps the network learn high-frequency patterns in spacetime.
    """
    # Create frequency multipliers: 2^0, 2^1, 2^2, ... 2^15
    freqs = torch.pow(2, torch.arange(num_encoding).float()).to(x.device)
    freqs = freqs.unsqueeze(0).unsqueeze(0)
    x_expanded = x.unsqueeze(2)
    # Apply sine and cosine at each frequency.
    sin_terms = torch.sin(freqs * x_expanded)
    cos_terms = torch.cos(freqs * x_expanded)
    # Concatenate sine terms, cosine terms, and original value for each input dimension.
    encoded = torch.cat([sin_terms, cos_terms, x_expanded], dim=2)
    return encoded.flatten(1, 2)

class StarNet(nn.Module):
    def __init__(self, num_blocks=4, in_dim=297, hidden_dim=256, inter_dim_factor=1,
                 drop_ratio=0.1, activation=F.softplus):
        super(StarNet, self).__init__()
        self.activation = activation
        # Input is (p0, v0, tau) -> 9 dims. Encoded -> 9 * (2*16 + 1) = 297
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.star_blocks = nn.ModuleList([
            StarBlock(hidden_dim, hidden_dim, inter_dim_factor, drop_ratio, activation)
            for _ in range(num_blocks)
        ])
        # Output is 4 correction terms for (t, r, theta, phi)
        self.fc2 = nn.Linear(hidden_dim, 4)
        self.init_weights()

    def forward(self, x, M=0.5):
        # x shape: [Batch, 9] -> [p0 (4), v0 (4), tau (1)]
        p0 = x[:, 0:4]
        v0 = x[:, 4:8]
        tau = x[:, 8:9]

        # Normalize time to prevent gradient explosion during training.
        tau_normalized = tau / 10.0 
        x_for_network = torch.cat([p0, v0, tau_normalized], dim=1)

        # Encode input with positional_encoding for better frequency learning.
        x_enc = positional_encoding(x_for_network)
        h = self.activation(self.fc1(x_enc))
        # Process through residual blocks.
        for blk in self.star_blocks:
            h = blk(h)
        # Get correction terms from the network.
        correction = self.fc2(h) 

        #Physics ansatz; position = initial + velocity * time + acceleration * time^2
        # This ensures the network satisfies initial conditions by construction.
        coords_raw = p0 + (v0 * tau) + (tau.pow(2) * correction)
        
        # Extract individual coordinates.
        t_raw   = coords_raw[:, 0:1]
        r_raw   = coords_raw[:, 1:2]
        th_raw  = coords_raw[:, 2:3]
        phi_raw = coords_raw[:, 3:4]
        
        # Enforce r > 0 to keep particles outside the event horizon.
        r = 1e-4 + F.softplus(r_raw)

        # Keep theta in valid range (0, π) to avoid singularities.
        theta = torch.clamp(th_raw, min=1e-3, max=math.pi-1e-3)
        
        # Return final coordinates.
        return torch.cat([t_raw, r, theta, phi_raw], dim=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)
        
        # Initialize final layer with tiny weights so the network starts with nearly flat space-time to prevent instability.
        nn.init.normal_(self.fc2.weight, mean=0.0, std=1e-5)
        nn.init.constant_(self.fc2.bias, 0.0)

def calculate_speed(network, init_p, direction, tau, M):
    """
    Compute position and velocity by differentiating network output w.r.t. time.
    Returns predicted position, velocity, and tau (with gradient tracking).
    """
    # Enable gradient computation for tau.
    tau = tau.clone().detach().requires_grad_(True)
    input_data = torch.cat([init_p, direction, tau], dim=1)
    output_pos = network(input_data, M)

    # Compute velocity as dx/dtau for each coordinate.
    vel_components = []
    for idx in range(output_pos.shape[1]):
        grad_component = torch.autograd.grad(
            output_pos[:, idx].sum(),
            tau,
            create_graph=True,
            retain_graph=True,
        )[0]
        vel_components.append(grad_component)
    dxtau = torch.cat(vel_components, dim=1)

    return output_pos, dxtau, tau

def calculate_speed_acc(network, init_p, direction, tau, M):
    """
    Compute position, velocity, and acceleration.
    Acceleration is the second derivative of position w.r.t. time.
    """
    output_pos, predict_v, tau_for_grad = calculate_speed(network, init_p, direction, tau, M)

    # Compute acceleration is dv/dtau for each velocity component.
    acc_components = []
    for idx in range(predict_v.shape[1]):
        grad_component = torch.autograd.grad(
            predict_v[:, idx].sum(),
            tau_for_grad,
            create_graph=True,
            retain_graph=True,
        )[0]
        acc_components.append(grad_component)
    predict_a = torch.cat(acc_components, dim=1)

    return output_pos, predict_v, predict_a

def get_christoffel_symbols(coords, M):
    """
    Calculate Christoffel symbols for Schwarzschild spacetime.
    These describe how spacetime is curved by the black hole's mass.
    """
    device = coords.device
    t, r, th, ph = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
    
    # Ensure r is safely outside the event horizon (r > 2M).
    r_safe = torch.clamp(r, min=2.01*M) 

    # Schwarzschild metric component and its derivative.
    f = 1.0 - 2.0 * M / r_safe
    df = 2.0 * M / (r_safe ** 2)
    sin_th = torch.sin(th)
    cos_th = torch.cos(th)
    sin2_th = sin_th ** 2
    cot_th = cos_th / (sin_th + 1e-12)

    # Christoffel symbols: Gamma^mu_ab tells how coordinates change in curved space.
    N = coords.shape[0]
    Gamma = torch.zeros((N, 4, 4, 4), device=device)

    # Time-related components (how time mixes with radio coordinate).
    val_ttr = df / (2.0 * f)
    Gamma[:, 0, 0, 1] = val_ttr
    Gamma[:, 0, 1, 0] = val_ttr

    # Radio components (how gravity affects radio motion).
    Gamma[:, 1, 0, 0] = f * df / 2.0
    Gamma[:, 1, 1, 1] = -df / (2.0 * f)
    Gamma[:, 1, 2, 2] = -r_safe * f
    Gamma[:, 1, 3, 3] = -r_safe * f * sin2_th

    # Angular theta components (spherical coordinate effects)
    one_over_r = 1.0 / r_safe
    Gamma[:, 2, 1, 2] = one_over_r
    Gamma[:, 2, 2, 1] = one_over_r
    Gamma[:, 2, 3, 3] = -sin_th * cos_th

    # Angular phi components (azmithal coordinate effects).
    Gamma[:, 3, 1, 3] = one_over_r
    Gamma[:, 3, 3, 1] = one_over_r
    Gamma[:, 3, 2, 3] = cot_th
    Gamma[:, 3, 3, 2] = cot_th

    return Gamma, None, None

def calculate_geodesic_residual_loss(predict_a, predict_v, input_coords, M):
    """
    Physics loss: particles should follow geodesics (straight paths in curved spacetime).
    Geodesic equation: d²x/dτ² + Γ(dx/dτ)(dx/dτ) = 0
    """
    # Get curvature effects from Christoffel symbols. 
    Gamma_mu_ab, _, _ = get_christoffel_symbols(input_coords, M)
    # Calculate how velocity curves due to space-time. 
    curvature_term = torch.einsum('nmab, na, nb -> nm', Gamma_mu_ab, predict_v, predict_v)
    # Geodesic_residual: should be zero if trajectory is correct. 
    geodesic_residual = predict_a + curvature_term
    # Minimize the residual to enforce physics. 
    physics_loss = F.mse_loss(geodesic_residual, torch.zeros_like(geodesic_residual))
    return physics_loss

def calculate_ic_loss(network, initial_p_batch, initial_v_batch, M):
    """
    Initial condition loss: at tau=0, network should output exact initial position/velocity.
    This is a data-driven constraint.
    """
    N = initial_p_batch.shape[0]
    tau_zeros = torch.zeros((N, 1), device=initial_p_batch.device)
    # Evaluate network at tau = 0. 
    output_pos, predict_v, _ = calculate_speed(network, initial_p_batch, initial_v_batch, tau_zeros, M)
    # Penalized deviation from initial conditions. 
    loss_pos_ic = F.mse_loss(output_pos, initial_p_batch)
    loss_vel_ic = F.mse_loss(predict_v, initial_v_batch)
    return loss_pos_ic + loss_vel_ic

def calculate_total_loss(network, physics_data, ic_data, M, lambda_phys=1.0, lambda_ic=1.0):
    """
    Combined loss: physics (geodesic equation) + initial conditions.
    Balances learning physics laws with matching training data.
    """
    init_p, direction, tau = physics_data
    # Compute position, velocity, and acceleration.
    output_pos, predict_v, predict_a = calculate_speed_acc(network, init_p, direction, tau, M)
    # Physics loss: enforced geodesic equation. 
    L_physics = calculate_geodesic_residual_loss(predict_a, predict_v, output_pos, M)
    # Data loss: match initial conditions. 
    L_data = calculate_ic_loss(network, ic_data['p0'], ic_data['v0'], M)
    # Weighted combination. 
    return (lambda_phys * L_physics) + (lambda_ic * L_data)

def sample_initial_conditions(batch_size, rs=1.0, device='cpu', dtype=torch.float32):
    """
    Generate random initial conditions for particles in strong gravity regime.
    Samples positions and velocities near the black hole.
    """
    # Sample radial positions in strong gravity zone (1.1 to 3.0 times Schwarzschild radius).
    r0 = torch.empty(batch_size, 1, device=device, dtype=dtype).uniform_(1.1 * rs, 3.0 * rs)
    # Random azimuthal angle. 
    phi0 = torch.empty(batch_size, 1, device=device, dtype=dtype).uniform_(0.0, 2.0 * math.pi)
    # Started Equatorial Plane. 
    theta0 = torch.full_like(r0, math.pi / 2.0)
    # Initial position: (t = 0, r, theta, phi) 
    p0 = torch.cat([torch.zeros_like(r0), r0, theta0, phi0], dim=1)

    # Sample velocity components. 
    vr = torch.empty(batch_size, 1, device=device, dtype=dtype).uniform_(-0.5, 0.5)
    vphi = torch.empty(batch_size, 1, device=device, dtype=dtype).uniform_(0.05, 0.15)
    vtheta = torch.zeros_like(vr)

    # Compute time component of velocity to ensure particles stay on mass shell. 
    # (Satisfy the constraint for mass of particles in general relativity.)
    f = 1.0 - rs / r0
    spatial = (1.0/f)*(vr**2) + (r0**2)*(vphi**2)
    vt = torch.sqrt((1.0 + spatial) / f)
    v0 = torch.cat([vt, vr, vtheta, vphi], dim=1)
    return p0, v0

def sample_collocation(batch_size, M=0.5, tau_range=(-0.5, 0.5), device='cpu', dtype=torch.float32):
    """
    Generate collocation points for training: random initial conditions + random times.
    Used to evaluate physics loss at various spacetime points.
    """
    rs = 2.0 * M # Schwarzschild radius 
    # Sample initial positions and velocities. 
    p0, v0 = sample_initial_conditions(batch_size, rs=rs, device=device, dtype=dtype)
    # Sample random proper time values. 
    tau = torch.empty(batch_size, 1, device=device, dtype=dtype).uniform_(*tau_range)
    return (p0, v0, tau), {'p0': p0, 'v0': v0}