import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import torch
import torch.optim as optim
from matplotlib.colors import LogNorm
from instrument_detection.training.adamw_optimiser import AdamW


def rosenbrock_3d(x: np.ndarray) -> float:
    """3D Rosenbrock function with global minimum at (1, 1, 1)."""
    x_val, y_val, z_val = x
    return ((1 - x_val) ** 2 + 100 * (y_val - x_val ** 2) ** 2 +
            (1 - z_val) ** 2 + 100 * (x_val - z_val ** 2) ** 2)


def rosenbrock_3d_grad(x: np.ndarray) -> np.ndarray:
    """Gradient of 3D Rosenbrock function."""
    x_val, y_val, z_val = x
    dx = -2 * (1 - x_val) - 400 * x_val * (y_val - x_val ** 2) + 200 * (x_val - z_val ** 2)
    dy = 200 * (y_val - x_val ** 2)
    dz = -2 * (1 - z_val) - 400 * z_val * (x_val - z_val ** 2)
    return np.array([dx, dy, dz])


def run_custom_adamw(init_params, num_steps, lr, weight_decay):
    """Run custom AdamW implementation."""
    params = [p.copy() for p in init_params]
    optimizer = AdamW(params, lr=lr, weight_decay=weight_decay)
    history = {'loss': [], 'params': []}
    for _ in range(num_steps):
        x = params[0]
        loss = rosenbrock_3d(x)
        grad = rosenbrock_3d_grad(x)
        history['loss'].append(loss)
        history['params'].append([p.copy() for p in params])
        optimizer.step([grad])
    return history


def run_pytorch_adamw(init_params, num_steps, lr, weight_decay):
    """Run PyTorch AdamW for comparison."""
    params = [torch.tensor(p.copy(), dtype=torch.float64, requires_grad=True) for p in init_params]
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    history = {'loss': [], 'params': []}
    for _ in range(num_steps):
        optimizer.zero_grad()
        x_np = params[0].detach().numpy()
        loss_val = rosenbrock_3d(x_np)
        grad = rosenbrock_3d_grad(x_np)
        params[0].grad = torch.tensor(grad, dtype=torch.float64)
        history['loss'].append(loss_val)
        history['params'].append([p.detach().numpy().copy() for p in params])
        optimizer.step()
    return history


def moving_average(data, window_size):
    """Computes the moving average of a 1D array."""
    return np.convolve(data, np.ones(window_size), 'valid') / window_size


def plot_relative_error(custom_history, pytorch_history, save_path='relative_error.pdf'):
    """Plots the smoothed relative error in loss between the two optimizers."""
    fig, ax = plt.subplots(figsize=(11, 7))
    custom_loss = np.array(custom_history['loss'])
    pytorch_loss = np.array(pytorch_history['loss'])

    epsilon = 1e-20
    relative_error = np.abs(custom_loss - pytorch_loss) / (np.abs(pytorch_loss) + epsilon)

    window_size = 25
    smoothed_error = moving_average(relative_error, window_size)
    smoothed_steps = range(window_size - 1, len(relative_error))

    ax.fill_between(smoothed_steps, smoothed_error, color='#d62728', alpha=0.8)
    ax.set_xlabel('Step', fontsize=24)
    ax.set_ylabel('Relative error', fontsize=24)

    # Enable grid for both major and minor ticks
    ax.grid(True, which='major', linestyle='--', alpha=0.5)
    ax.grid(True, which='minor', linestyle=':', alpha=0.4)  # This line adds the intermediate grid

    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    print(f"Plot saved to {save_path}")
    plt.show()


def plot_parameter_evolution(custom_history, pytorch_history, save_path='parameter_evolution.pdf'):
    """Plots the evolution of each parameter for both optimizers."""
    fig, ax = plt.subplots(figsize=(11, 7))
    custom_params = np.array([p[0] for p in custom_history['params']])
    pytorch_params = np.array([p[0] for p in pytorch_history['params']])
    steps = range(len(custom_params))
    param_labels = ['X', 'Y', 'Z']

    lines = []
    labels = []
    for i in range(3):
        line1, = ax.plot(steps, custom_params[:, i], linestyle='-', label=f'Custom Implementation {param_labels[i]}')
        line2, = ax.plot(steps, pytorch_params[:, i], linestyle='--', label=f'PyTorch Implementation {param_labels[i]}')
        lines.extend([line1, line2])
        labels.extend([f'Custom {param_labels[i]}', f'PyTorch {param_labels[i]}'])

    ax.set_xlabel('Step', fontsize=24)
    ax.set_ylabel('Parameter Value', fontsize=24)
    ax.legend(handles=lines, labels=labels, ncol=3, loc='lower right', fontsize=18)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.tick_params(axis='both', which='major', labelsize=24)

    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    print(f"Plot saved to {save_path}")
    plt.show()


def plot_3d_loss_surface_trajectory(custom_history, pytorch_history, save_path='trajectory_loss_surface_3d.pdf'):
    """Plots the 3D trajectory of the optimizers on the loss surface."""
    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection='3d')

    custom_params = np.array([p[0] for p in custom_history['params']])
    pytorch_params = np.array([p[0] for p in pytorch_history['params']])
    custom_loss = np.array(custom_history['loss'])
    pytorch_loss = np.array(pytorch_history['loss'])

    x_range = np.linspace(-1.5, 1.5, 100)
    y_range = np.linspace(-1.0, 3.0, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z_loss = rosenbrock_3d(np.array([X, Y, np.ones_like(X)]))

    ax.plot_surface(X, Y, Z_loss, cmap=cm.viridis, alpha=0.8, rstride=5, cstride=5, edgecolor='none')
    ax.plot(custom_params[:, 0], custom_params[:, 1], custom_loss, color='#2ca02c', linewidth=2.5,
            label='Custom Trajectory')
    ax.plot(pytorch_params[:, 0], pytorch_params[:, 1], pytorch_loss, color='red', linestyle='--', linewidth=2.5,
            label='PyTorch Trajectory')
    ax.scatter(custom_params[0, 0], custom_params[0, 1], custom_loss[0], c='white', edgecolors='black', s=100,
               marker='o', label='Start Point', zorder=5)
    ax.scatter(1, 1, 0, c='black', s=150, marker='o', label='Global minimum', zorder=5)

    ax.set_xlabel('X Parameter', fontsize=18)
    ax.set_ylabel('Y Parameter', fontsize=18)
    ax.set_zlabel('Loss', fontsize=18)
    ax.view_init(elev=40, azim=-40)
    ax.legend(fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    print(f"Plot saved to {save_path}")
    plt.show()


def plot_2d_rosenbrock_trajectory(custom_history, pytorch_history, save_path='rosenbrock_trajectory_2d.pdf'):
    """Plots the 2D trajectory of the optimizers on the Rosenbrock function contour."""
    fig, ax = plt.subplots(figsize=(11, 7))

    custom_params = np.array([p[0] for p in custom_history['params']])
    pytorch_params = np.array([p[0] for p in pytorch_history['params']])

    x_range = np.linspace(-1.5, 1.5, 200)
    y_range = np.linspace(-1.0, 3.0, 200)
    X, Y = np.meshgrid(x_range, y_range)

    Z_loss = rosenbrock_3d(np.array([X, Y, np.ones_like(X)]))
    levels = np.logspace(0, 5, 35)
    norm = LogNorm()
    Z_loss_clipped = np.maximum(Z_loss, 1.0)

    contour = ax.contourf(X, Y, Z_loss_clipped, levels=levels, cmap=cm.viridis, norm=norm)
    cbar = fig.colorbar(contour)

    # CORRECTED SECTION: Set explicit, clean ticks at powers of 10
    tick_locations = [1, 10, 100, 1000, 10000]
    cbar.set_ticks(tick_locations)

    cbar.set_label('Loss (Log Scale)', size=20)
    cbar.ax.tick_params(labelsize=24)

    ax.contour(X, Y, Z_loss, levels=levels, colors='white', linewidths=0.5, alpha=0.3, norm=norm)

    ax.plot(custom_params[:, 0], custom_params[:, 1], color='lime', linewidth=3, linestyle='-',
            label='Custom Trajectory', zorder=3)
    ax.plot(pytorch_params[:, 0], pytorch_params[:, 1], color='red', linestyle='--', linewidth=3,
            label='PyTorch Trajectory', zorder=3)

    ax.scatter(custom_params[0, 0], custom_params[0, 1], c='white', edgecolors='black', s=120, marker='o',
               label='Start Point', zorder=5)

    ax.scatter(1, 1, c='black', edgecolors='white', s=150, marker='o', label='Global Minimum', zorder=5)

    ax.set_xlabel('X Parameter', fontsize=24)
    ax.set_ylabel('Y Parameter', fontsize=24)
    ax.legend(fontsize=18)
    ax.grid(True, linestyle='--', color='white', alpha=0.2)
    ax.tick_params(axis='both', which='major', labelsize=24)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.0, 3.0)

    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    print(f"Plot saved to {save_path}")
    plt.show()


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    lr = 0.01
    weight_decay = 0.01
    num_steps = 1000
    init_params = [np.array([-1.0, 2.0, -0.5])]

    print("Running Custom AdamW...")
    custom_history = run_custom_adamw(init_params, num_steps, lr, weight_decay)

    print("Running PyTorch AdamW...")
    pytorch_history = run_pytorch_adamw(init_params, num_steps, lr, weight_decay)

    print("\nGenerating final plots...")
    # Call the individual plotting functions
    plot_relative_error(custom_history, pytorch_history)
    plot_parameter_evolution(custom_history, pytorch_history)
    plot_3d_loss_surface_trajectory(custom_history, pytorch_history)
    plot_2d_rosenbrock_trajectory(custom_history, pytorch_history)


if __name__ == "__main__":
    main()