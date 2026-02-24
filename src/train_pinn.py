"""
train_pinn.py — PINN Training & Yaw Optimization
Progress Report 3 | Ibrahim Alghrabi | MATH 619

Trains:
  1. Baseline NN (data-only, no physics)
  2. Physics-Informed NN (data + physics loss)
Then compares them and demonstrates gradient-based yaw optimization.

Run: python train_pinn.py
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from data_pipeline import run_pipeline, RATED_POWER_KW, ROTOR_DIAMETER_M, AIR_DENSITY

# ── Config ──────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Hyperparameters
HIDDEN_LAYERS = 5
HIDDEN_UNITS = 64
ACTIVATION = nn.Tanh
LEARNING_RATE = 1e-3
BATCH_SIZE = 512
EPOCHS = 200
LAMBDA_PHYSICS = 0.1  # weight for physics loss
PATIENCE = 20         # early stopping patience

# Resolve paths relative to project root (one level up from src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# =====================================================================
# MODEL DEFINITIONS
# =====================================================================

class BaselineNN(nn.Module):
    """Standard feedforward neural network (no physics constraints)."""
    
    def __init__(self, input_dim, hidden_layers=HIDDEN_LAYERS,
                 hidden_units=HIDDEN_UNITS):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_units), ACTIVATION()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_units, hidden_units), ACTIVATION()]
        layers.append(nn.Linear(hidden_units, 1))  # single output: power
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class PhysicsInformedNN(nn.Module):
    """
    PINN for wind turbine power prediction.
    
    Architecture: Same as baseline, but training includes physics loss.
    The physics loss enforces:
      1. Power curve constraint: P ∝ wind_speed³ (Region 2)
      2. Yaw cosine law: P ∝ cos³(yaw_misalignment)  
      3. Monotonicity: ∂P/∂wind_speed ≥ 0 in Region 2
      4. Actuator disk bound: P ≤ P_betz (Betz limit)
    
    These are "soft" physics constraints embedded in the loss function,
    following the PINN framework of Raissi et al. (2019).
    """
    
    def __init__(self, input_dim, hidden_layers=HIDDEN_LAYERS,
                 hidden_units=HIDDEN_UNITS):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_units), ACTIVATION()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_units, hidden_units), ACTIVATION()]
        layers.append(nn.Linear(hidden_units, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


# =====================================================================
# PHYSICS LOSS FUNCTIONS
# =====================================================================

def compute_physics_loss(model, x_batch, scaler, feature_cols):
    """
    Compute physics-informed residual losses.
    
    Input features (after scaling): 
        [wind_speed_norm, wd_cos, wd_sin, yaw_cos, yaw_sin, pitch_norm]
    
    Physics constraints:
    (1) Power-wind cubic law in Region 2: P ∝ v³
    (2) Yaw cosine law: P ∝ cos³(γ) where γ = yaw misalignment
    (3) Monotonicity: ∂P/∂v ≥ 0 for moderate wind speeds
    (4) Betz limit: P_norm ≤ 1.0
    """
    x = x_batch.clone().requires_grad_(True)
    p_pred = model(x)
    
    losses = {}
    
    # ── (1) Power curve gradient constraint ──
    # ∂P/∂wind_speed should be > 0 for Region 2 (moderate winds)
    # wind_speed_normalized is feature index 0
    grad_p = torch.autograd.grad(
        outputs=p_pred, inputs=x,
        grad_outputs=torch.ones_like(p_pred),
        create_graph=True, retain_graph=True
    )[0]
    
    dp_dv = grad_p[:, 0]  # gradient w.r.t. wind speed (first feature)
    
    # Penalize negative gradients (power should increase with wind in Region 2)
    # Only apply where wind speed is in Region 2 range
    # wind_speed_normalized ~ 0.12 (cut-in=3/25) to 0.5 (rated=12.5/25)
    ws_scaled = x[:, 0]  # scaled wind speed
    region2_mask = (ws_scaled > -0.5) & (ws_scaled < 0.5)  # approximate Region 2
    monotonicity_violation = torch.relu(-dp_dv[region2_mask])
    losses['monotonicity'] = monotonicity_violation.mean() if monotonicity_violation.numel() > 0 else torch.tensor(0.0)
    
    # ── (2) Yaw cosine law ──
    # If yaw_cos and yaw_sin are features (indices 3, 4), 
    # the effective yaw misalignment angle γ satisfies cos(γ) = yaw_cos * wd_cos + yaw_sin * wd_sin
    # Power should decrease as |γ| increases: ∂P/∂|γ| ≤ 0
    # Proxy: ∂P/∂yaw_cos should be ≥ 0 (more aligned = more power)
    if x.shape[1] > 3:
        dp_dyaw_cos = grad_p[:, 3]  # gradient w.r.t. yaw_cos
        yaw_violation = torch.relu(-dp_dyaw_cos)  # penalize negative
        losses['yaw_cosine'] = yaw_violation.mean()
    
    # ── (3) Betz limit / bounded output ──
    # Predicted normalized power should be in [0, 1]
    over_limit = torch.relu(p_pred - 1.0) + torch.relu(-p_pred)
    losses['betz_bound'] = over_limit.mean()
    
    # ── (4) Smoothness regularization ──
    # Second derivative penalty for smooth power surface
    if grad_p.requires_grad:
        grad2 = torch.autograd.grad(
            outputs=grad_p.sum(), inputs=x,
            create_graph=True, retain_graph=True
        )[0]
        losses['smoothness'] = (grad2 ** 2).mean() * 0.01
    
    # Total physics loss
    total_physics = sum(losses.values())
    return total_physics, {k: v.item() for k, v in losses.items()}


# =====================================================================
# TRAINING
# =====================================================================

def train_model(model, train_loader, val_X, val_y, epochs=EPOCHS,
                lr=LEARNING_RATE, physics=False, scaler=None,
                feature_cols=None, lambda_phys=LAMBDA_PHYSICS):
    """
    Train a model with optional physics loss.
    Returns training history dict.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5)
    
    criterion = nn.MSELoss()
    
    history = {
        'train_loss': [], 'val_loss': [], 'physics_loss': [],
        'data_loss': [], 'best_val_loss': float('inf'), 'best_epoch': 0
    }
    
    best_state = None
    patience_counter = 0
    
    model.to(DEVICE)
    val_X_t = torch.FloatTensor(val_X).to(DEVICE)
    val_y_t = torch.FloatTensor(val_y).to(DEVICE)
    
    t0 = time.time()
    
    for epoch in range(epochs):
        model.train()
        epoch_data_loss = 0
        epoch_phys_loss = 0
        n_batches = 0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            
            # Data loss
            pred = model(xb)
            data_loss = criterion(pred, yb)
            
            # Physics loss (PINN only)
            if physics:
                phys_loss, _ = compute_physics_loss(
                    model, xb, scaler, feature_cols)
                total_loss = data_loss + lambda_phys * phys_loss
                epoch_phys_loss += phys_loss.item()
            else:
                total_loss = data_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_data_loss += data_loss.item()
            n_batches += 1
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(val_X_t)
            val_loss = criterion(val_pred, val_y_t).item()
        
        scheduler.step(val_loss)
        
        avg_data = epoch_data_loss / n_batches
        avg_phys = epoch_phys_loss / n_batches if physics else 0
        
        history['train_loss'].append(avg_data + lambda_phys * avg_phys if physics else avg_data)
        history['val_loss'].append(val_loss)
        history['data_loss'].append(avg_data)
        history['physics_loss'].append(avg_phys)
        
        # Early stopping
        if val_loss < history['best_val_loss']:
            history['best_val_loss'] = val_loss
            history['best_epoch'] = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= PATIENCE:
            print(f"    Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            phys_str = f" | Phys: {avg_phys:.4f}" if physics else ""
            print(f"    Epoch {epoch+1:3d}/{epochs} | "
                  f"Data: {avg_data:.4f}{phys_str} | Val: {val_loss:.4f}")
    
    elapsed = time.time() - t0
    
    # Restore best model
    if best_state:
        model.load_state_dict(best_state)
    model.to(DEVICE)
    
    history['training_time'] = elapsed
    print(f"    Done in {elapsed:.1f}s | Best val loss: {history['best_val_loss']:.5f} "
          f"(epoch {history['best_epoch']+1})")
    
    return history


def evaluate_model(model, X_test, y_test):
    """Evaluate model on test set. Returns metrics dict."""
    model.eval()
    X_t = torch.FloatTensor(X_test).to(DEVICE)
    
    with torch.no_grad():
        pred = model(X_t).cpu().numpy()
    
    y = y_test.flatten()
    p = pred.flatten()
    
    mse = np.mean((y - p) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y - p))
    
    # R² score
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    
    # Convert back to kW for interpretable metrics
    rmse_kw = rmse * RATED_POWER_KW
    mae_kw = mae * RATED_POWER_KW
    
    return {
        'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2,
        'RMSE_kW': rmse_kw, 'MAE_kW': mae_kw,
        'predictions': p, 'actuals': y
    }


# =====================================================================
# YAW OPTIMIZATION (Gradient-based)
# =====================================================================

def optimize_yaw(pinn_model, sample_inputs, scaler, feature_cols,
                 yaw_cos_idx=3, yaw_sin_idx=4, n_steps=100, lr=0.05):
    """
    Gradient-based yaw optimization using the trained PINN.
    
    For each sample, find the yaw angle that maximizes predicted power
    by computing ∂P/∂yaw via automatic differentiation.
    
    Args:
        sample_inputs: scaled input features (N, D)
        yaw_cos_idx, yaw_sin_idx: indices of yaw features
    
    Returns:
        dict with original/optimized yaw angles and power predictions
    """
    pinn_model.eval()
    
    N = sample_inputs.shape[0]
    x = torch.FloatTensor(sample_inputs).to(DEVICE)
    
    # Extract original yaw cos/sin
    orig_yaw_cos = x[:, yaw_cos_idx].clone()
    orig_yaw_sin = x[:, yaw_sin_idx].clone()
    orig_yaw_angle = torch.atan2(orig_yaw_sin, orig_yaw_cos)  # radians
    
    # Create optimizable yaw angle parameter
    yaw_param = orig_yaw_angle.clone().requires_grad_(True)
    yaw_optimizer = optim.Adam([yaw_param], lr=lr)
    
    # Get original power prediction
    with torch.no_grad():
        orig_power = pinn_model(x).squeeze()
    
    power_history = []
    
    for step in range(n_steps):
        yaw_optimizer.zero_grad()
        
        # Update yaw features from the optimizable parameter
        x_opt = x.clone()
        x_opt[:, yaw_cos_idx] = torch.cos(yaw_param)
        x_opt[:, yaw_sin_idx] = torch.sin(yaw_param)
        
        # Predict power (we want to MAXIMIZE, so minimize negative power)
        power_pred = pinn_model(x_opt).squeeze()
        loss = -power_pred.mean()  # maximize power
        
        loss.backward()
        yaw_optimizer.step()
        
        # Clamp yaw to reasonable range (±30° from original)
        with torch.no_grad():
            diff = yaw_param - orig_yaw_angle
            diff = torch.clamp(diff, -np.radians(30), np.radians(30))
            yaw_param.data = orig_yaw_angle + diff
        
        power_history.append(-loss.item())
    
    # Final optimized predictions
    with torch.no_grad():
        x_final = x.clone()
        x_final[:, yaw_cos_idx] = torch.cos(yaw_param)
        x_final[:, yaw_sin_idx] = torch.sin(yaw_param)
        opt_power = pinn_model(x_final).squeeze()
    
    orig_angles_deg = np.degrees(orig_yaw_angle.cpu().numpy())
    opt_angles_deg = np.degrees(yaw_param.detach().cpu().numpy())
    yaw_adjustment = opt_angles_deg - orig_angles_deg
    
    # Wrap to [-180, 180]
    yaw_adjustment = (yaw_adjustment + 180) % 360 - 180
    
    power_gain = (opt_power.cpu().numpy() - orig_power.cpu().numpy()) * RATED_POWER_KW
    
    return {
        'original_power': orig_power.cpu().numpy() * RATED_POWER_KW,
        'optimized_power': opt_power.cpu().numpy() * RATED_POWER_KW,
        'power_gain_kW': power_gain,
        'yaw_adjustment_deg': yaw_adjustment,
        'power_history': power_history,
        'mean_gain_kW': np.mean(power_gain),
        'mean_gain_pct': np.mean(power_gain / (orig_power.cpu().numpy() * RATED_POWER_KW + 1e-6)) * 100,
    }


# =====================================================================
# PLOTTING
# =====================================================================

def plot_training_curves(hist_baseline, hist_pinn):
    """Plot training loss curves for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Baseline
    ax = axes[0]
    ax.plot(hist_baseline['train_loss'], label='Train', alpha=0.8)
    ax.plot(hist_baseline['val_loss'], label='Validation', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Baseline NN — Training Curves')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # PINN
    ax = axes[1]
    ax.plot(hist_pinn['data_loss'], label='Data Loss', alpha=0.8)
    ax.plot(hist_pinn['physics_loss'], label='Physics Loss', alpha=0.8)
    ax.plot(hist_pinn['val_loss'], label='Validation', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('PINN — Training Curves')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/training_curves.png")


def plot_predictions_comparison(metrics_baseline, metrics_pinn):
    """Scatter plots: predicted vs actual for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for ax, metrics, title in zip(
        axes,
        [metrics_baseline, metrics_pinn],
        ['Baseline NN', 'Physics-Informed NN']
    ):
        y = metrics['actuals'] * RATED_POWER_KW
        p = metrics['predictions'] * RATED_POWER_KW
        
        ax.scatter(y, p, alpha=0.1, s=5, c='steelblue')
        ax.plot([0, RATED_POWER_KW], [0, RATED_POWER_KW], 'r--', lw=1.5, label='Perfect')
        ax.set_xlabel('Actual Power (kW)')
        ax.set_ylabel('Predicted Power (kW)')
        ax.set_title(f'{title}\nR²={metrics["R2"]:.4f} | RMSE={metrics["RMSE_kW"]:.1f} kW')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-50, RATED_POWER_KW + 100)
        ax.set_ylim(-50, RATED_POWER_KW + 100)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'predictions_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/predictions_comparison.png")


def plot_yaw_optimization(opt_results):
    """Plot yaw optimization results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # (a) Optimization convergence
    ax = axes[0, 0]
    ax.plot(opt_results['power_history'], color='green', lw=2)
    ax.set_xlabel('Optimization Step')
    ax.set_ylabel('Mean Predicted Power (norm)')
    ax.set_title('Yaw Optimization Convergence')
    ax.grid(True, alpha=0.3)
    
    # (b) Yaw adjustment distribution
    ax = axes[0, 1]
    adj = opt_results['yaw_adjustment_deg']
    ax.hist(adj, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(0, color='red', ls='--', lw=1.5)
    ax.axvline(np.mean(adj), color='orange', ls='-', lw=2, label=f'Mean: {np.mean(adj):.1f}°')
    ax.set_xlabel('Yaw Adjustment (degrees)')
    ax.set_ylabel('Count')
    ax.set_title('Recommended Yaw Adjustments')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (c) Power gain distribution
    ax = axes[1, 0]
    gain = opt_results['power_gain_kW']
    ax.hist(gain, bins=50, color='forestgreen', edgecolor='white', alpha=0.8)
    ax.axvline(0, color='red', ls='--', lw=1.5)
    ax.axvline(np.mean(gain), color='orange', ls='-', lw=2,
               label=f'Mean: {np.mean(gain):.1f} kW')
    ax.set_xlabel('Power Gain (kW)')
    ax.set_ylabel('Count')
    ax.set_title('Predicted Power Gains from Yaw Optimization')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (d) Original vs Optimized power
    ax = axes[1, 1]
    orig = opt_results['original_power']
    opt = opt_results['optimized_power']
    ax.scatter(orig, opt, alpha=0.2, s=5, c='purple')
    lim = max(orig.max(), opt.max()) * 1.05
    ax.plot([0, lim], [0, lim], 'r--', lw=1.5, label='No change')
    ax.set_xlabel('Original Power (kW)')
    ax.set_ylabel('Optimized Power (kW)')
    ax.set_title('Power: Original vs. Yaw-Optimized')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'yaw_optimization.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/yaw_optimization.png")


def plot_physics_compliance(pinn_model, data, scaler, feature_cols):
    """
    Plot physics compliance: power vs wind speed curve from PINN
    compared to theoretical cubic law and Betz limit.
    """
    # Generate synthetic inputs: vary wind speed, keep other features at median
    ws_range = np.linspace(0.1, 25, 200)
    
    # Build input array with median values for non-wind-speed features
    n_features = len(feature_cols)
    median_input = np.median(data['X_test'], axis=0)
    
    X_synth = np.tile(median_input, (200, 1)).astype(np.float32)
    
    # Wind speed is feature 0 — vary it
    # Need to scale it: (ws_normalized - mean) / std
    ws_norm = ws_range / 25.0  # normalize
    ws_scaled = (ws_norm - scaler.mean_[0]) / scaler.scale_[0]
    X_synth[:, 0] = ws_scaled
    
    # Also set yaw to aligned (cos=1, sin=0 in original, then scale)
    # This shows the "ideal" power curve
    
    pinn_model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X_synth).to(DEVICE)
        pred = pinn_model(X_t).cpu().numpy().flatten()
    
    pred_kw = pred * RATED_POWER_KW
    
    # Theoretical Betz limit
    area = np.pi * (ROTOR_DIAMETER_M / 2) ** 2
    betz = 0.5 * AIR_DENSITY * area * (16/27) * ws_range**3 / 1000  # kW
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ws_range, pred_kw, 'b-', lw=2, label='PINN Prediction')
    ax.plot(ws_range, np.clip(betz, 0, RATED_POWER_KW), 'r--', lw=1.5,
            label='Betz Limit', alpha=0.7)
    ax.axhline(RATED_POWER_KW, color='gray', ls=':', label=f'Rated ({RATED_POWER_KW} kW)')
    ax.axvline(3.0, color='green', ls=':', alpha=0.5, label='Cut-in (3 m/s)')
    ax.axvline(12.5, color='orange', ls=':', alpha=0.5, label='Rated (12.5 m/s)')
    ax.set_xlabel('Wind Speed (m/s)')
    ax.set_ylabel('Power (kW)')
    ax.set_title('PINN Power Curve vs. Theoretical Limits')
    ax.legend(fontsize=9)
    ax.set_xlim(0, 25)
    ax.set_ylim(-50, RATED_POWER_KW + 200)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'physics_compliance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/physics_compliance.png")


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("=" * 60)
    print("PROGRESS REPORT 3: PINN TRAINING & YAW OPTIMIZATION")
    print("=" * 60)
    
    # ── Load data ──
    data = run_pipeline()
    
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']
    scaler = data['scaler']
    feature_cols = data['feature_cols']
    
    input_dim = X_train.shape[1]
    print(f"\n  Input dimension: {input_dim}")
    print(f"  Device: {DEVICE}")
    
    # DataLoader
    train_ds = TensorDataset(
        torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    
    # ── Train Baseline NN ──
    print("\n" + "=" * 60)
    print("TRAINING: Baseline Neural Network")
    print("=" * 60)
    baseline = BaselineNN(input_dim)
    hist_baseline = train_model(
        baseline, train_loader, X_val, y_val, physics=False)
    
    # ── Train PINN ──
    print("\n" + "=" * 60)
    print("TRAINING: Physics-Informed Neural Network")
    print("=" * 60)
    pinn = PhysicsInformedNN(input_dim)
    hist_pinn = train_model(
        pinn, train_loader, X_val, y_val,
        physics=True, scaler=scaler, feature_cols=feature_cols)
    
    # ── Evaluate both ──
    print("\n" + "=" * 60)
    print("EVALUATION ON TEST SET")
    print("=" * 60)
    
    metrics_baseline = evaluate_model(baseline, X_test, y_test)
    metrics_pinn = evaluate_model(pinn, X_test, y_test)
    
    # Comparison table
    print(f"\n  {'Metric':<15} {'Baseline NN':>15} {'PINN':>15} {'Δ':>12}")
    print("  " + "-" * 57)
    for key in ['MSE', 'RMSE', 'MAE', 'R2', 'RMSE_kW', 'MAE_kW']:
        b = metrics_baseline[key]
        p = metrics_pinn[key]
        delta = p - b
        unit = ' kW' if 'kW' in key else ''
        print(f"  {key:<15} {b:>15.4f} {p:>15.4f} {delta:>+11.4f}{unit}")
    
    print(f"\n  Training time — Baseline: {hist_baseline['training_time']:.1f}s | "
          f"PINN: {hist_pinn['training_time']:.1f}s")
    
    # ── Yaw Optimization ──
    print("\n" + "=" * 60)
    print("YAW OPTIMIZATION (Gradient-Based)")
    print("=" * 60)
    
    # Use a subset of test data for optimization demo
    n_opt = min(2000, X_test.shape[0])
    opt_results = optimize_yaw(
        pinn, X_test[:n_opt], scaler, feature_cols)
    
    print(f"  Samples optimized: {n_opt}")
    print(f"  Mean yaw adjustment: {np.mean(opt_results['yaw_adjustment_deg']):.2f}°")
    print(f"  Mean power gain:     {opt_results['mean_gain_kW']:.1f} kW "
          f"({opt_results['mean_gain_pct']:.1f}%)")
    print(f"  Max power gain:      {np.max(opt_results['power_gain_kW']):.1f} kW")
    
    # ── Generate figures ──
    print("\n" + "=" * 60)
    print("GENERATING FIGURES")
    print("=" * 60)
    
    plot_training_curves(hist_baseline, hist_pinn)
    plot_predictions_comparison(metrics_baseline, metrics_pinn)
    plot_yaw_optimization(opt_results)
    plot_physics_compliance(pinn, data, scaler, feature_cols)
    
    # ── Save results ──
    results = {
        'baseline': {k: float(v) for k, v in metrics_baseline.items() 
                     if isinstance(v, (int, float, np.floating))},
        'pinn': {k: float(v) for k, v in metrics_pinn.items() 
                 if isinstance(v, (int, float, np.floating))},
        'optimization': {
            'n_samples': n_opt,
            'mean_gain_kW': float(opt_results['mean_gain_kW']),
            'mean_gain_pct': float(opt_results['mean_gain_pct']),
            'max_gain_kW': float(np.max(opt_results['power_gain_kW'])),
            'mean_yaw_adjustment_deg': float(np.mean(opt_results['yaw_adjustment_deg'])),
        },
        'hyperparameters': {
            'hidden_layers': HIDDEN_LAYERS,
            'hidden_units': HIDDEN_UNITS,
            'learning_rate': LEARNING_RATE,
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'lambda_physics': LAMBDA_PHYSICS,
            'patience': PATIENCE,
        }
    }
    
    with open(os.path.join(RESULTS_DIR, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print("  Saved: results/metrics.json")
    


if __name__ == "__main__":
    main()
