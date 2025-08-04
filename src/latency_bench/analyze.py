import pandas as pd
import argparse
import numpy as np

def analyze_latency(file_path, baseline_device):
    """
    Analyzes benchmark data from a CSV file to compare latency across different devices.

    For each unique combination of model, batch size, and sequence length, this function
    calculates the percentage increase or decrease in latency for other devices
    compared to a specified baseline device. It also provides a summary for
    high-intensity workloads.

    Args:
        file_path (str): The path to the input CSV file.
        baseline_device (str): The device_type to use as the baseline (e.g., 'cpu').
    """
    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return

    # Get a list of all unique devices to compare against the baseline
    all_devices = df['device_type'].unique()
    comparison_devices = [d for d in all_devices if d != baseline_device]
    
    # Store results for the final summary
    summary_comparisons = []

    # Group the data by the desired columns
    grouped = df.groupby(['model', 'batch_size', 'seq_len'])

    print(f"--- Latency Performance Comparison (vs. {baseline_device.upper()}) ---")

    # Iterate over each group
    for (model, batch_size, seq_len), group in grouped:
        print(f"\nModel: {model}, Batch Size: {batch_size}, Sequence Length: {seq_len}")

        try:
            # Find the baseline latency
            baseline_latency = group[group['device_type'] == baseline_device]['latency_ms'].iloc[0]
            print(f"  - {baseline_device.upper()} Latency: {baseline_latency:.2f} ms (Baseline)")

            # Compare each other device to the baseline
            for device in comparison_devices:
                device_group = group[group['device_type'] == device]
                if not device_group.empty:
                    device_latency = device_group['latency_ms'].iloc[0]
                    percentage_change = ((device_latency - baseline_latency) / baseline_latency) * 100
                    change_type = "slower" if percentage_change > 0 else "faster"
                    print(f"  - {device.upper()} Latency: {device_latency:.2f} ms ({abs(percentage_change):.2f}% {change_type})")
                    
                    # Store data for the final summary
                    summary_comparisons.append({
                        'batch_size': batch_size,
                        'seq_len': seq_len,
                        'comparison_device': device,
                        'percent_change': percentage_change
                    })
                else:
                    print(f"  - {device.upper()} Latency: Not available")

        except IndexError:
            print(f"  - Could not find {baseline_device.upper()} baseline for this group. Skipping comparison.")

    # --- Final Summary Section ---
    if not summary_comparisons:
        print("\n--- No data available for summary ---")
        return
        
    summary_df = pd.DataFrame(summary_comparisons)
    
    # Filter for high-intensity workloads
    high_intensity_df = summary_df[
        (summary_df['seq_len'] >= 256) & 
        (summary_df['batch_size'] >= 512)
    ]

    print("\n\n--- Summary for High-Intensity Workloads ---")
    print("(Sequence Length > 256 and Batch Size > 512)")

    if high_intensity_df.empty:
        print("No data points matched the high-intensity criteria.")
    else:
        # Calculate the average percentage change for each device
        summary_results = high_intensity_df.groupby('comparison_device')['percent_change'].mean()
        
        for device, avg_change in summary_results.items():
            change_type = "slower" if avg_change > 0 else "faster"
            print(f"\nAverage performance of {device.upper()} vs. {baseline_device.upper()}:")
            print(f"  - On average, {abs(avg_change):.2f}% {change_type}")


if __name__ == '__main__':
    # Set up the argument parser to accept a command-line argument
    parser = argparse.ArgumentParser(
        description="Analyze benchmark data from a CSV to compare latency across devices."
    )
    # Add a positional argument for the CSV file path
    parser.add_argument(
        "csv_file",
        type=str,
        help="The path to the input CSV file."
    )
    # Add an optional argument for the baseline device
    parser.add_argument(
        "--baseline",
        type=str,
        default="cpu",
        help="The device_type to use as the baseline for comparison (e.g., cpu, mps, inf2)."
    )
    # Parse the arguments from the command line
    args = parser.parse_args()
    
    # Call the analysis function with the file path and baseline provided by the user
    analyze_latency(args.csv_file, args.baseline)