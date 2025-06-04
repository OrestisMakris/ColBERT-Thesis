import os
import torch
import torch.nn.functional as F
import argparse
from glob import glob
import random
import time
import matplotlib.pyplot as plt
import numpy as np

# --- Fixed Configuration ---
FIXED_INPUT_DIR_NAME = "qd_matrices_all_pairs"
FIXED_OUTPUT_DIR_NAME = "padded_matrices_cnn"
FIXED_PADDING_VALUE = 0.0
# --- End Fixed Configuration ---

def print_message(message):
    """Prints a message with a timestamp."""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

def pad_matrix_torch(tensor, target_shape, padding_value):
    """
    Pads or truncates a 2D tensor to a target shape.
    target_shape is (target_height, target_width)
    """
    current_shape = tensor.shape
    target_height, target_width = target_shape

    pad_bottom = max(0, target_height - current_shape[0])
    pad_right = max(0, target_width - current_shape[1])
    
    # F.pad format for 2D: (pad_left, pad_right, pad_top, pad_bottom)
    padded_tensor = F.pad(tensor, (0, pad_right, 0, pad_bottom), mode='constant', value=padding_value)
    
    # Truncate if current dimensions were larger
    padded_tensor = padded_tensor[:target_height, :target_width]
    return padded_tensor

def main():
    parser = argparse.ArgumentParser(description="Pad exported .pt similarity matrices to automatically determined max dimensions and save them. Optionally plots heatmaps.")
    parser.add_argument("--plot_padded_heatmaps", action='store_true', 
                        help="Generate and save a heatmap for one randomly selected padded matrix per unique query ID.")
    args = parser.parse_args()

    base_dir = os.getcwd()
    input_dir = os.path.join(base_dir, FIXED_INPUT_DIR_NAME)
    output_dir = os.path.join(base_dir, FIXED_OUTPUT_DIR_NAME)

    os.makedirs(output_dir, exist_ok=True)
    
    print_message(f"Input directory (fixed): {input_dir}")
    print_message(f"Output directory for padded .pt files (fixed): {output_dir}")
    print_message(f"Padding value (fixed): {FIXED_PADDING_VALUE}")

    unpadded_pt_files = glob(os.path.join(input_dir, "q*_d*.pt"))
    if not unpadded_pt_files:
        print_message(f"No .pt files found in the input directory: {input_dir}. Exiting.")
        return
    print_message(f"Found {len(unpadded_pt_files)} .pt files to process from {input_dir}.")

    max_q_len_found = 0
    max_d_len_found = 0
    valid_files_for_dim_check = []

    print_message("Scanning all input files to determine maximum query and document dimensions...")
    for pt_file_path in unpadded_pt_files:
        try:
            tensor = torch.load(pt_file_path, map_location='cpu')
            if tensor.ndim == 2:
                q_len, d_len = tensor.shape
                max_q_len_found = max(max_q_len_found, q_len)
                max_d_len_found = max(max_d_len_found, d_len)
                valid_files_for_dim_check.append(pt_file_path)
            else:
                print_message(f"Warning: File {pt_file_path} does not contain a 2D tensor. Shape: {tensor.shape}. Skipping.")
        except Exception as e:
            print_message(f"Error loading or checking dimensions for {pt_file_path}: {e}. Skipping.")
    
    if not valid_files_for_dim_check:
        print_message("No valid 2D tensor .pt files found to determine dimensions or pad. Exiting.")
        return

    unpadded_pt_files = valid_files_for_dim_check 

    final_cnn_max_query_len = max_q_len_found if max_q_len_found > 0 else 1
    final_cnn_max_doc_len = max_d_len_found if max_d_len_found > 0 else 1
    
    print_message(f"Determined Max Query Length from all valid files: {final_cnn_max_query_len}")
    print_message(f"Determined Max Document Length from all valid files: {final_cnn_max_doc_len}")
    print_message(f"All valid matrices will be padded/truncated to shape: ({final_cnn_max_query_len}, {final_cnn_max_doc_len})")
    
    processed_count = 0
    total_original_size = 0
    total_padded_size = 0
    
    print_message(f"Starting serial padding for {len(unpadded_pt_files)} files...")

    for i, pt_file_path in enumerate(unpadded_pt_files):
        filename = os.path.basename(pt_file_path)
        output_pt_filepath = os.path.join(output_dir, filename)
        
        try:
            original_tensor = torch.load(pt_file_path, map_location='cpu')
            if original_tensor.ndim != 2:
                continue

            original_size = os.path.getsize(pt_file_path)
            original_shape = original_tensor.shape
            
            padded_tensor = pad_matrix_torch(original_tensor, 
                                             (final_cnn_max_query_len, final_cnn_max_doc_len), 
                                             FIXED_PADDING_VALUE)
            
            torch.save(padded_tensor, output_pt_filepath)
            padded_size = os.path.getsize(output_pt_filepath)
            
            processed_count += 1
            total_original_size += original_size
            total_padded_size += padded_size
            
            if (processed_count % 1000 == 0) or (processed_count == len(unpadded_pt_files)):
                print_message(f"Processed {processed_count}/{len(unpadded_pt_files)} files. Last: {filename} Original shape: {original_shape} -> Padded shape: {padded_tensor.shape}")

        except Exception as e:
            print_message(f"Error processing file {filename}: {e}")


    print_message(f"Padding complete. Successfully processed {processed_count} .pt files.")
    print_message(f"Total size of original matrices (bytes): {total_original_size}")
    print_message(f"Total size of padded matrices (bytes): {total_padded_size}")
    print_message(f"Padded matrices saved to: {output_dir}")

    if args.plot_padded_heatmaps and processed_count > 0:
        plot_output_dir = os.path.join(output_dir, "random_padded_query_heatmaps")
        os.makedirs(plot_output_dir, exist_ok=True)
        print_message(f"Attempting to plot random padded heatmaps to: {plot_output_dir}")

        padded_pt_files_for_plotting = glob(os.path.join(output_dir, "q*_d*.pt"))
        
        query_files_map = {}
        for padded_file_path in padded_pt_files_for_plotting:
            filename = os.path.basename(padded_file_path)
            try:
                query_id_str = filename.split('_')[0][1:] 
                if query_id_str not in query_files_map:
                    query_files_map[query_id_str] = []
                query_files_map[query_id_str].append(padded_file_path)
            except IndexError:
                print_message(f"Could not parse query ID from filename: {filename}. Skipping for plotting.")
                continue
        
        plotted_count = 0
        if not query_files_map:
            print_message("No padded .pt files found in output directory.")
        
        for query_id, files_for_query in query_files_map.items():
            if not files_for_query:
                continue
            
            randomly_selected_padded_pt_path = random.choice(files_for_query)
            selected_filename_no_ext = os.path.splitext(os.path.basename(randomly_selected_padded_pt_path))[0]
            plot_png_filepath = os.path.join(plot_output_dir, f"q{query_id}_random_padded.png")

            try:
                padded_tensor = torch.load(randomly_selected_padded_pt_path, map_location='cpu')
                
                # Use matplotlib to create a heatmap
                plt.figure(figsize=(max(5, padded_tensor.shape[1] * 0.05), max(4, padded_tensor.shape[0] * 0.05)))
                
                # Plot the heatmap with viridis colormap (default, good for similarity data)
                im = plt.imshow(padded_tensor.numpy(), aspect="auto", cmap="viridis")
                
                # Add colorbar to show the mapping of colors to values
                plt.colorbar(label="Similarity Score")
                
                # Add labels and title
                plt.xlabel(f"Document token position (Padded to {final_cnn_max_doc_len})")
                plt.ylabel(f"Query token position (Padded to {final_cnn_max_query_len})")
                plt.title(f"Random Padded Matrix for Query {query_id}\n(Source: {selected_filename_no_ext}.pt)")
                
                # Save the plot with tight layout
                plt.tight_layout()
                plt.savefig(plot_png_filepath, dpi=100, bbox_inches="tight")
                plt.close()
                
                plotted_count += 1
                print_message(f"Plotted heatmap for Query ID {query_id} to {plot_png_filepath}")
            except Exception as e:
                print_message(f"Error plotting heatmap for Query ID {query_id}: {e}")
        
        if plotted_count > 0:
            print_message(f"Plotted {plotted_count} heatmaps to: {plot_output_dir}")
        else:
            print_message("No heatmaps were plotted.")

if __name__ == "__main__":
    main()