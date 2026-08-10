import torch
from torch.cuda.amp import autocast

def gradient_analysis(model):
    print("\nGradient Analysis:")
    print("-" * 80)

    total_norm_squared = 0
    params_with_grad = 0
    zero_grad_params = 0

    layer_info = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_data = param.grad.data
            grad_abs = grad_data.abs()

            grad_norm = grad_data.norm(2).item()
            grad_mean = grad_abs.mean().item()
            grad_max = grad_abs.max().item()
            grad_min = grad_abs.min().item()
            zero_ratio = (grad_abs < 1e-10).float().mean().item()

            total_norm_squared += grad_norm ** 2
            params_with_grad += 1

            if zero_ratio > 0.9:
                status = "Mostly zero"
            elif grad_mean < 1e-8:
                status = "Very small"
            elif grad_mean > 1.0:
                status = "Very large"
            else:
                status = "Normal"

            if grad_mean < 1e-8:
                zero_grad_params += 1

            layer_info.append({
                'name': name,
                'norm': grad_norm,
                'mean': grad_mean,
                'max': grad_max,
                'min': grad_min,
                'zero_ratio': zero_ratio,
                'status': status,
                'param_count': grad_data.numel()
            })

            print(f"{name:>25}: norm={grad_norm:>8.2e}, mean={grad_mean:>8.2e}, "
                  f"max={grad_max:>8.2e}, zeros={zero_ratio:>5.1%} {status}")
        else:
            print(f"{name:>25}: No gradient")

    total_norm = total_norm_squared ** 0.5

    print(f"\nGradient Summary:")
    print(f"Total gradient norm: {total_norm:.6f}")
    print(f"Parameters with gradients: {params_with_grad}")
    print(f"Parameters with near zero gradients: {zero_grad_params}")
    print(f"Percentage with gradient issues: {100 * zero_grad_params / params_with_grad:.1f}%")

    if total_norm < 1e-8:
        print("WARNING: Gradients are essentially zero.")

    return total_norm, layer_info

def check_model_health(model, data_loader, device, verbose=True):
    if verbose:
        print("-" * 80)
        print("Model health check:")

    model.eval()

    try:
        with torch.no_grad():
            batch_data = next(iter(data_loader))
            data, target, _, _ = batch_data
            data, target = data.to(device), target.to(device)

            if verbose:
                print(f"Batch shape: {data.shape}")
                print(f"Target shape: {target.shape}")

            with autocast():
                output = model(data)

            outputs = {'output': output}

            for name, tensor in outputs.items():
                if torch.isnan(tensor).any():
                    print(f"CRITICAL: NaN detected in {name}!")
                    return False

                if torch.isinf(tensor).any():
                    print(f"CRITICAL: Infinite values detected in {name}!")
                    return False

                abs_max = tensor.abs().max().item()
                if abs_max > 100:
                    print(f"WARNING: Extreme values detected in {name} (max: {abs_max:.2f})")
                    if abs_max > 1000:
                        return False

                if verbose:
                    stats = {
                        'mean': tensor.mean().item(),
                        'std': tensor.std().item(),
                        'min': tensor.min().item(),
                        'max': tensor.max().item()
                    }

                    print(f"{name.capitalize()} statistics:")
                    print(f"  Mean: {stats['mean']:.6f}")
                    print(f"  Std: {stats['std']:.6f}")
                    print(f"  Range: [{stats['min']:.6f}, {stats['max']:.6f}]")

                    if abs(stats['mean']) > 10:
                        print(f"  WARNING: {name} mean is large")
                    if stats['std'] > 10:
                        print(f"  WARNING: {name} std is large")

            if verbose and hasattr(model, 'print_activation_stats'):
                model.print_activation_stats()
            if verbose:
                print("Model health check passed!")
            return True

    except Exception as e:
        print(f"ERROR during model health check: {e}")
        return False

def print_cache_summary(datasets):
    print(f"\n{'=' * 60}")
    print(f"CACHE USAGE SUMMARY")
    print(f"{'=' * 60}")

    total_cache_size = 0

    for name, dataset in datasets.items():
        cache_info = dataset.get_cache_info()
        if isinstance(cache_info, dict):
            print(f"\n{name} Dataset:")
            print(f"  Cache directory: {cache_info['cache_dir']}")
            print(f"  HCQT cache files: {len(cache_info['hcqt_files'])}")
            print(f"  Metadata cache files: {len(cache_info['metadata_files'])}")
            print(f"  Total size: {cache_info['total_size_mb']:.1f} MB")
            total_cache_size += cache_info['total_size_mb']

    print(f"\nOverall cache usage: {total_cache_size:.1f} MB")

def print_results(metrics_dict, instrument_labels):
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    print("\nPer-Instrument Metrics:")
    header = f"{'Instrument':<15} | {'Precision':>10} | {'Recall':>10} | {'F1-Score':>10}"
    print(header)
    print("-" * len(header))

    per_class = metrics_dict.get('per_class', {})
    precisions = per_class.get('precision', [])
    recalls = per_class.get('recall', [])
    f1s = per_class.get('f1', [])

    for i, label in enumerate(instrument_labels):
        p = precisions[i] if i < len(precisions) else -1
        r = recalls[i] if i < len(recalls) else -1
        f1 = f1s[i] if i < len(f1s) else -1
        print(f"{label:<15} | {p:>10.4f} | {r:>10.4f} | {f1:>10.4f}")

    print("="*60 + "\n")

    print("\nGlobal Metrics:")
    print(f"  - Macro F1-Score:     {metrics_dict['macro']['f1']:.4f}")
    print(f"  - Macro Precision:    {metrics_dict['macro']['precision']:.4f}")
    print(f"  - Macro Recall:       {metrics_dict['macro']['recall']:.4f}")
    print(f"  - Frame Accuracy:     {metrics_dict['frame_accuracy']:.4f}")
    print(f"  - Hamming Loss:       {metrics_dict['hamming_loss']:.4f}")
