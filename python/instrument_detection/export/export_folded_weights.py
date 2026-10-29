import torch
import torch.nn as nn
import os
import numpy as np
from instrument_detection.models.model import build_model
from instrument_detection.config import *

# --- CONFIGURATION ---
# Path to the model you trained (with the InstanceNorm layer)
MODEL_PATH = 'models/instrument_model_2025-10-16_13-42-05.pth'
# Where the C engine looks for them (c/assets/c_weights, when run from c/)
OUTPUT_DIR = '../c/assets/c_weights'


def save_tensor_to_file(tensor, sub_dir, filename):
    """Helper function to save a tensor to a binary file."""
    target_dir = os.path.join(OUTPUT_DIR, sub_dir)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    filepath = os.path.join(target_dir, filename)
    tensor.detach().cpu().numpy().astype(np.float32).tofile(filepath)
    print(f"  -> Exported to {filepath}")


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    print(f"--- Starting BatchNorm Folding and Weight Export ---")
    print(f"Output directory: '{OUTPUT_DIR}'")

    device = torch.device("cpu")
    model = build_model()

    print(f"\nLoading model from: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set model to evaluation mode

    # --- FOLDING AND EXPORTING LOGIC ---

    # 1. Handle the initial InstanceNorm layer (no folding, just copy)
    print("\nProcessing 'input_norm' layer...")
    instancenorm_layer = model.input_norm
    save_tensor_to_file(instancenorm_layer.weight, 'input_norm', 'weight.bin')
    save_tensor_to_file(instancenorm_layer.bias, 'input_norm', 'bias.bin')

    # 2. Iterate through Conv-BN pairs in the backbone and heads
    components_to_fold = {
        "backbone": model.backbone,
        "piano_aggregator": model.piano_aggregator,
        "strings_aggregator": model.strings_aggregator,
        "woodwinds_aggregator": model.woodwinds_aggregator,
        "brass_aggregator": model.brass_aggregator
    }

    for name, component in components_to_fold.items():
        print(f"\nProcessing '{name}' component...")
        # We need to find sequential Conv2d -> BatchNorm2d pairs
        module_list = list(component.modules())
        for i in range(len(module_list) - 1):
            current_module = module_list[i]
            next_module = module_list[i + 1]

            if isinstance(current_module, nn.Conv2d) and isinstance(next_module, nn.BatchNorm2d):
                conv = current_module
                bn = next_module

                # Get original parameters
                conv_weight = conv.weight
                # The backbone convs have no bias, so we start with a zero tensor
                conv_bias = conv.bias if conv.bias is not None else torch.zeros(conv.out_channels, device=device)

                bn_mean = bn.running_mean
                bn_var = bn.running_var
                bn_gamma = bn.weight
                bn_beta = bn.bias
                epsilon = bn.eps

                # --- The Math for Folding ---
                scale = bn_gamma / torch.sqrt(bn_var + epsilon)

                # Fold the scale into the conv weights and bias
                folded_weight = conv_weight * scale.view(-1, 1, 1, 1)
                folded_bias = (conv_bias - bn_mean) * scale + bn_beta

                # Find the original layer name (e.g., '0', '3' for backbone)
                layer_name = [n for n, m in component.named_modules() if m is conv][0]

                print(f"  - Folding Conv layer '{layer_name}' with its BatchNorm...")
                # Save the new, folded parameters
                sub_dir = name.split('_')[0]  # 'backbone', 'piano', etc.
                save_tensor_to_file(folded_weight, sub_dir, f"{layer_name}_folded_weight.bin")
                save_tensor_to_file(folded_bias, sub_dir, f"{layer_name}_folded_bias.bin")

    # 3. Handle the final classifier layers (no folding, just copy)
    print("\nProcessing final classifier layers...")
    for head_name in ['piano', 'strings', 'woodwinds', 'brass']:
        classifier = getattr(model, f"{head_name}_classifier")
        print(f"  - Exporting '{head_name}_classifier' weights...")
        save_tensor_to_file(classifier.weight, head_name, "classifier_weight.bin")
        save_tensor_to_file(classifier.bias, head_name, "classifier_bias.bin")

    print(f"\n--- SUCCESS: All parameters folded and exported. ---")


if __name__ == '__main__':
    main()