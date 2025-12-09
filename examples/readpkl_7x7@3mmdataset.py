import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy arrays and complex numbers"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.complexfloating):
            return {'real': float(obj.real), 'imag': float(obj.imag), '_complex': True}
        elif isinstance(obj, complex):
            return {'real': obj.real, 'imag': obj.imag, '_complex': True}
        return super(NumpyEncoder, self).default(obj)

def convert_numpy_to_lists(obj):
    """Recursively convert numpy arrays and complex numbers to JSON-serializable types"""
    if isinstance(obj, dict):
        return {key: convert_numpy_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_lists(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, (np.complexfloating, complex)):
        return {'real': float(obj.real), 'imag': float(obj.imag), '_complex': True}
    else:
        return obj

def load_pickle_file(pkl_path):
    """Load data from pickle file"""
    try:
        with open(pkl_path, 'rb') as file:
            data = pickle.load(file)
        print(f"Successfully loaded pickle file: {pkl_path}")
        return data
    except Exception as e:
        print(f"Error loading pickle file: {str(e)}")
        return None

def convert_pkl_to_json(pkl_path, json_path=None):
    """Convert pickle file to JSON"""
    # Load pickle data
    data = load_pickle_file(pkl_path)
    if data is None:
        return False
    
    # Generate JSON path if not provided
    if json_path is None:
        base_name = os.path.splitext(pkl_path)[0]
        json_path = f"{base_name}_converted.json"
    
    try:
        # Convert to JSON-serializable format
        json_data = convert_numpy_to_lists(data)
        
        # Save to JSON
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2, cls=NumpyEncoder)
        
        print(f"Successfully converted to JSON: {json_path}")
        return True
    except Exception as e:
        print(f"Error converting to JSON: {str(e)}")
        return False

def analyze_simulation_data(data):
    """Analyze the simulation data and provide summary statistics"""
    print("\n" + "="*60)
    print("SIMULATION DATA ANALYSIS")
    print("="*60)
    
    # Metadata analysis
    metadata = data.get('metadata', {})
    print(f"Number of matrices: {metadata.get('num_matrices', 'N/A')}")
    print(f"Matrix size: {metadata.get('matrix_size', 'N/A')}")
    print(f"Generation date: {metadata.get('generation_date', 'N/A')}")
    print(f"Total simulation time: {metadata.get('total_simulation_time_minutes', 'N/A'):.2f} minutes")
    
    # Simulation results analysis
    simulations = data.get('simulations', [])
    successful_sims = [sim for sim in simulations if sim.get('success', False)]
    failed_sims = [sim for sim in simulations if not sim.get('success', False)]
    
    print(f"\nSimulation Results:")
    print(f"- Total simulations: {len(simulations)}")
    print(f"- Successful: {len(successful_sims)}")
    print(f"- Failed: {len(failed_sims)}")
    print(f"- Success rate: {len(successful_sims)/len(simulations)*100:.1f}%")
    
    if successful_sims:
        sim_times = [sim.get('simulation_time_seconds', 0) for sim in successful_sims]
        print(f"\nSimulation Times:")
        print(f"- Average: {np.mean(sim_times):.2f} seconds")
        print(f"- Min: {np.min(sim_times):.2f} seconds")
        print(f"- Max: {np.max(sim_times):.2f} seconds")
    
    return successful_sims, failed_sims

def create_matrix_visualization(data, save_plots=True):
    """Create visualizations of the matrices and results"""
    simulations = data.get('simulations', [])
    successful_sims = [sim for sim in simulations if sim.get('success', False)]
    
    if not successful_sims:
        print("No successful simulations to visualize")
        return
    
    # Create output directory for plots
    if save_plots:
        plot_dir = "visualization_plots"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
    
    # 1. Matrix pattern visualization (first 16 matrices)
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle('First 16 Matrix Patterns', fontsize=16)
    
    for i, ax in enumerate(axes.flat):
        if i < len(successful_sims) and i < 16:
            matrix = np.array(successful_sims[i]['matrix'])
            ax.imshow(matrix, cmap='binary', interpolation='nearest')
            ax.set_title(f'Matrix {i+1}')
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis('off')
    
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(plot_dir, 'matrix_patterns.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. TE values distribution
    te_values = []
    for sim in successful_sims:
        te = sim.get('te', [])
        if te and isinstance(te, list) and len(te) > 0:
            # Take the first value or average if multiple
            te_val = te[0] if isinstance(te[0], (int, float)) else np.mean([abs(x) for x in te])
            te_values.append(te_val)
    
    if te_values:
        plt.figure(figsize=(10, 6))
        plt.hist(te_values, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('TE Values')
        plt.ylabel('Frequency')
        plt.title('Distribution of TE Values')
        plt.grid(True, alpha=0.3)
        if save_plots:
            plt.savefig(os.path.join(plot_dir, 'te_distribution.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    # 3. Simulation time analysis
    sim_times = [sim.get('simulation_time_seconds', 0) for sim in successful_sims]
    if sim_times:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(sim_times, 'b-', alpha=0.7)
        plt.xlabel('Simulation Number')
        plt.ylabel('Time (seconds)')
        plt.title('Simulation Time vs Simulation Number')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.hist(sim_times, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Simulation Time (seconds)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Simulation Times')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(os.path.join(plot_dir, 'simulation_times.png'), dpi=300, bbox_inches='tight')
        plt.show()

def create_dataframe_summary(data):
    """Create pandas DataFrame for easy data manipulation"""
    simulations = data.get('simulations', [])
    successful_sims = [sim for sim in simulations if sim.get('success', False)]
    
    if not successful_sims:
        print("No successful simulations to create DataFrame")
        return None
    
    # Extract key information
    summary_data = []
    for sim in successful_sims:
        matrix = np.array(sim['matrix'])
        
        # Matrix statistics
        matrix_density = np.mean(matrix)  # Proportion of 1s
        matrix_sum = np.sum(matrix)
        
        # TE statistics
        te = sim.get('te', [])
        te_mean = np.mean(te) if te else 0
        te_std = np.std(te) if te and len(te) > 1 else 0
        
        summary_data.append({
            'matrix_id': sim['matrix_id'],
            'matrix_density': matrix_density,
            'matrix_sum': matrix_sum,
            'te_mean': te_mean,
            'te_std': te_std,
            'simulation_time': sim.get('simulation_time_seconds', 0),
            'success': sim.get('success', False)
        })
    
    df = pd.DataFrame(summary_data)
    print("\nDataFrame Summary:")
    print(df.describe())
    
    return df

def main():
    """Main function to convert and visualize pickle file"""
    # Your pickle file path
    pkl_path = r"C:\Users\shivam\cst-python-api\examples\batch_simulation_results\cst_batch_results_20250613_153723.pkl"
    
    print("Loading and analyzing pickle file...")
    
    # Load the data
    data = load_pickle_file(pkl_path)
    if data is None:
        return
    
    # Convert to JSON
    print("\nConverting to JSON...")
    convert_pkl_to_json(pkl_path)
    
    # Analyze the data
    successful_sims, failed_sims = analyze_simulation_data(data)
    
    # Create DataFrame summary
    df = create_dataframe_summary(data)
    if df is not None:
        # Save DataFrame to CSV
        csv_path = pkl_path.replace('.pkl', '_summary.csv')
        df.to_csv(csv_path, index=False)
        print(f"\nDataFrame saved to: {csv_path}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_matrix_visualization(data)
    
    # Print some sample data
    print("\n" + "="*60)
    print("SAMPLE DATA")
    print("="*60)
    
    if successful_sims:
        sample_sim = successful_sims[0]
        print(f"Sample Matrix (ID: {sample_sim['matrix_id']}):")
        matrix = np.array(sample_sim['matrix'])
        print(matrix)
        print(f"\nTE values: {sample_sim.get('te', 'N/A')}")
        print(f"Frequency: {sample_sim.get('frequency', 'N/A')}")
        print(f"Simulation time: {sample_sim.get('simulation_time_seconds', 'N/A')} seconds")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()