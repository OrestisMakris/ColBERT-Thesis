# import torch
# import matplotlib.pyplot as plt
# import argparse
# import os
# import time

# #DEFAULT_PT_FILE_PATH = "/home/st1084516/ColBERT-Thesis/qd_matrices_fiqa_untuned/q22_d15644.pt"

# # --- Fixed Configuration ---
# # Change this to the specific .pt file you want to visualize by default
# DEFAULT_PT_FILE_PATH = "/home/st1084516/ColBERT-Thesis/qd_matrices_fiqa_untuned/q110_d9273.pt"
# #DEFAULT_PT_FILE_PATH = "/home/st1084516/ColBERT-Thesis/qd_matrices_fiqa_untuned/q4_d39513.pt"
# # --- End Fixed Configuration ---

# def print_message(message):
#     """Prints a message with a timestamp."""
#     print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

# def main():
#     # Parse command line arguments
#     parser = argparse.ArgumentParser(description="Create a heatmap from a PyTorch tensor file (.pt)")
#     parser.add_argument("--input_file", default=DEFAULT_PT_FILE_PATH, 
#                         help=f"Path to the input .pt file (default: {DEFAULT_PT_FILE_PATH})")
#     parser.add_argument("--cmap", default="viridis", help="Matplotlib colormap to use (default: viridis)")
#     parser.add_argument("--dpi", type=int, default=150, help="DPI for saved image (default: 150)")
#     parser.add_argument("--output_dir", default=None, 
#                         help="Output directory (default: same directory as the input file)")
#     args = parser.parse_args()

#     # Validate input file
#     if not os.path.exists(args.input_file):
#         print_message(f"Error: Input file '{args.input_file}' does not exist.")
#         return

#     # Determine output path - same folder as input, but with .png extension
#     input_path = os.path.abspath(args.input_file)
#     input_dir = os.path.dirname(input_path)
#     input_filename = os.path.basename(input_path)
#     output_filename = "/home/st1084516/ColBERT-Thesis/colbert_run" "_heatmap80--.png"
    
#     # Use specified output directory if provided, otherwise use input directory
#     output_dir = args.output_dir if args.output_dir else input_dir
#     output_path = os.path.join(output_dir, output_filename)

#     # Load the tensor
#     try:
#         print_message(f"Loading tensor from {args.input_file}...")
#         tensor = torch.load(args.input_file, map_location='cpu')
#         print(tensor)
#         print_message(f"Loaded tensor of shape {tensor.shape}")
#     except Exception as e:
#         print_message(f"Error loading tensor: {e}")
#         return

#     # Verify it's a 2D tensor
#     if tensor.ndim != 2:
#         print_message(f"Error: Expected a 2D tensor, but got shape {tensor.shape}")
#         return

#     print_message(f"Creating heatmap visualization...")
    
#     # Calculate figure size based on tensor dimensions
#     # This ensures that larger tensors get larger figures
#     q_len, d_len = tensor.shape
    
#     # Base size factors - adjust these to change the overall size
#     width_factor = 0.05  # inches per column
#     height_factor = 0.05  # inches per row
    
#     # Minimum size to ensure small tensors are still visible
#     min_width = 6
#     min_height = 5
    
#     # Calculate dimensions while maintaining reasonable proportions
#     fig_width = max(min_width, d_len * width_factor)
#     fig_height = max(min_height, q_len * height_factor)
    
#     print_message(f"Using figure size of {fig_width:.1f} x {fig_height:.1f} inches")
    
#     # Create the heatmap figure with dynamic size
#     plt.figure(figsize=(fig_width, fig_height))
    
#     # Plot the tensor as a heatmap
#     im = plt.imshow(tensor.numpy(), cmap=args.cmap, aspect='auto')
    
#     # Add colorbar and labels
#     plt.colorbar(label="Value")
#     plt.title(f"Heatmap of {input_filename}")
#     plt.xlabel("Column Index (Document Tokens)")
#     plt.ylabel("Row Index (Query Tokens)")
    
#     # Add statistics
#     tensor_np = tensor.numpy()
#     min_val = tensor_np.min()
#     max_val = tensor_np.max()
#     mean_val = tensor_np.mean()
    
#     # Check for padding value (-100.0 is often used in ColBERT)
#     has_padding = (tensor_np == -100.0).any()
#     if has_padding:
#         # Count percentage of padding
#         padding_count = (tensor_np == -100.0).sum()
#         total_elements = tensor_np.size
#         padding_percentage = (padding_count / total_elements) * 100
        
#         # Add text about padding
#         plt.figtext(0.5, 0.01, 
#                    f"Contains padding (-100.0): {padding_percentage:.1f}% of elements\n"
#                    f"Stats: Min={min_val:.3f}, Max={max_val:.3f}, Mean={mean_val:.3f}",
#                    ha="center", fontsize=8, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
#     else:
#         plt.figtext(0.5, 0.01, 
#                    f"Stats: Min={min_val:.3f}, Max={max_val:.3f}, Mean={mean_val:.3f}",
#                    ha="center", fontsize=8, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    
#     # Save the figure
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
#     print_message(f"Saved heatmap to {output_path}")

# if __name__ == "__main__":
#     main()

import torch
import matplotlib.pyplot as plt
import os
import time


INPUT_FILE = "/home/st1084516/ColBERT-Thesis/qd_matrices_fiqa_untuned/q114_d1120.pt"
OUTPUT_DIR = "/home/st1084516/ColBERT-Thesis/images/heatmaps"

CMAP = "viridis"  # Colormap (e.g., "viridis", "plasma", "inferno", "magma", "cividis")
DPI = 150         # Dots Per Inch for the saved image file


def print_message(message):
    """Prints a message with a timestamp."""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

# --- Main script logic ---

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Validate input file
if not os.path.exists(INPUT_FILE):
    print_message(f"Error: Input file '{INPUT_FILE}' does not exist.")
    exit()

# Determine the output path based on the input filename
base_filename = os.path.splitext(os.path.basename(INPUT_FILE))[0]
output_filename = f"_heatmap.png"
output_path = os.path.join(OUTPUT_DIR, output_filename)

# Load the tensor
try:
    print_message(f"Loading tensor from {INPUT_FILE}...")
    tensor = torch.load(INPUT_FILE, map_location='cpu')
    print_message(f"Loaded tensor of shape {tensor.shape}")
except Exception as e:
    print_message(f"Error loading tensor: {e}")
    exit()

# Verify it's a 2D tensor
if tensor.ndim != 2:
    print_message(f"Error: Expected a 2D tensor, but got shape {tensor.shape}")
    exit()

print_message(f"Creating heatmap visualization...")

# Calculate figure size based on tensor dimensions
q_len, d_len = tensor.shape
width_factor = 0.05
height_factor = 0.05
min_width = 6
min_height = 5
fig_width = max(min_width, d_len * width_factor)
fig_height = max(min_height, q_len * height_factor)

print_message(f"Using figure size of {fig_width:.1f} x {fig_height:.1f} inches")

plt.figure(figsize=(fig_width, fig_height))

im = plt.imshow(tensor.numpy(), cmap=CMAP, aspect='auto', vmin=0.0, vmax=1.0)

# Add colorbar and labels
plt.colorbar(im, label="Similarity Score")
plt.title(f"Heatmap of {os.path.basename(INPUT_FILE)}")
plt.xlabel("Document Tokens")
plt.ylabel("Query Tokens")

# Add statistics text box
tensor_np = tensor.numpy()
min_val = tensor_np.min()
max_val = tensor_np.max()
mean_val = tensor_np.mean()

stats_text = f"Stats: Min={min_val:.3f}, Max={max_val:.3f}, Mean={mean_val:.3f}"

plt.figtext(0.5, 0.01, stats_text,
           ha="center", fontsize=8, bbox={"facecolor":"white", "alpha":0.5, "pad":5})

# Save the figure
plt.tight_layout(rect=[0, 0.05, 1, 1]) # Adjust layout to make space for figtext
plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
print_message(f"Saved heatmap to {output_path}")