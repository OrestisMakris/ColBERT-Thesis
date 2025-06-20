import torch
import matplotlib.pyplot as plt
import argparse
import os
import time

# --- Fixed Configuration ---
# Change this to the specific .pt file you want to visualize by default
DEFAULT_PT_FILE_PATH = "/home/st1084516/ColBERT-Thesis/qd_matrices_beir_untuned/q0_d1000.pt"
# --- End Fixed Configuration ---

def print_message(message):
    """Prints a message with a timestamp."""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Create a heatmap from a PyTorch tensor file (.pt)")
    parser.add_argument("--input_file", default=DEFAULT_PT_FILE_PATH, 
                        help=f"Path to the input .pt file (default: {DEFAULT_PT_FILE_PATH})")
    parser.add_argument("--cmap", default="viridis", help="Matplotlib colormap to use (default: viridis)")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for saved image (default: 150)")
    parser.add_argument("--output_dir", default=None, 
                        help="Output directory (default: same directory as the input file)")
    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.input_file):
        print_message(f"Error: Input file '{args.input_file}' does not exist.")
        return

    # Determine output path - same folder as input, but with .png extension
    input_path = os.path.abspath(args.input_file)
    input_dir = os.path.dirname(input_path)
    input_filename = os.path.basename(input_path)
    output_filename = "/home/st1084516/ColBERT-Thesis/colbert_run" "_heatmap80--.png"
    
    # Use specified output directory if provided, otherwise use input directory
    output_dir = args.output_dir if args.output_dir else input_dir
    output_path = os.path.join(output_dir, output_filename)

    # Load the tensor
    try:
        print_message(f"Loading tensor from {args.input_file}...")
        tensor = torch.load(args.input_file, map_location='cpu')
        print(tensor)
        print_message(f"Loaded tensor of shape {tensor.shape}")
    except Exception as e:
        print_message(f"Error loading tensor: {e}")
        return

    # Verify it's a 2D tensor
    if tensor.ndim != 2:
        print_message(f"Error: Expected a 2D tensor, but got shape {tensor.shape}")
        return

    print_message(f"Creating heatmap visualization...")
    
    # Calculate figure size based on tensor dimensions
    # This ensures that larger tensors get larger figures
    q_len, d_len = tensor.shape
    
    # Base size factors - adjust these to change the overall size
    width_factor = 0.05  # inches per column
    height_factor = 0.05  # inches per row
    
    # Minimum size to ensure small tensors are still visible
    min_width = 6
    min_height = 5
    
    # Calculate dimensions while maintaining reasonable proportions
    fig_width = max(min_width, d_len * width_factor)
    fig_height = max(min_height, q_len * height_factor)
    
    print_message(f"Using figure size of {fig_width:.1f} x {fig_height:.1f} inches")
    
    # Create the heatmap figure with dynamic size
    plt.figure(figsize=(fig_width, fig_height))
    
    # Plot the tensor as a heatmap
    im = plt.imshow(tensor.numpy(), cmap=args.cmap, aspect='auto')
    
    # Add colorbar and labels
    plt.colorbar(label="Value")
    plt.title(f"Heatmap of {input_filename}")
    plt.xlabel("Column Index (Document Tokens)")
    plt.ylabel("Row Index (Query Tokens)")
    
    # Add statistics
    tensor_np = tensor.numpy()
    min_val = tensor_np.min()
    max_val = tensor_np.max()
    mean_val = tensor_np.mean()
    
    # Check for padding value (-100.0 is often used in ColBERT)
    has_padding = (tensor_np == -100.0).any()
    if has_padding:
        # Count percentage of padding
        padding_count = (tensor_np == -100.0).sum()
        total_elements = tensor_np.size
        padding_percentage = (padding_count / total_elements) * 100
        
        # Add text about padding
        plt.figtext(0.5, 0.01, 
                   f"Contains padding (-100.0): {padding_percentage:.1f}% of elements\n"
                   f"Stats: Min={min_val:.3f}, Max={max_val:.3f}, Mean={mean_val:.3f}",
                   ha="center", fontsize=8, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    else:
        plt.figtext(0.5, 0.01, 
                   f"Stats: Min={min_val:.3f}, Max={max_val:.3f}, Mean={mean_val:.3f}",
                   ha="center", fontsize=8, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    print_message(f"Saved heatmap to {output_path}")

if __name__ == "__main__":
    main()