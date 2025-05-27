"""
Simplified script to run the Habitor pipeline
"""
import os
import sys

# Add src directory to path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.main import HabitorPipeline
    
    print("Starting Habitor Pipeline")
    
    # Create pipeline with dataset
    pipeline = HabitorPipeline(data_path="student_habits_dataset.csv", 
                               output_dir="output", 
                               classification_mode=False)
    
    # Run selected steps
    print("Step 1: Data Collection")
    pipeline.collect_data()
    
    print("Step 2: Data Visualization")
    pipeline.visualize_data()
    
    print("Pipeline steps completed")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc() 