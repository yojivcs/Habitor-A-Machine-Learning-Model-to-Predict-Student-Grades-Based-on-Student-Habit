import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

class DataCollector:
    def __init__(self, data_path):
        """
        Initialize the DataCollector with the path to the dataset
        
        Args:
            data_path (str): Path to the student habits dataset
        """
        self.data_path = data_path
        self.data = None
        
    def load_data(self):
        """
        Load the dataset from the provided path
        
        Returns:
            pandas.DataFrame: The loaded dataset
        """
        print(f"Loading data from {self.data_path}")
        try:
            # Check the file extension and load accordingly
            file_extension = os.path.splitext(self.data_path)[1].lower()
            
            if file_extension == '.csv':
                self.data = pd.read_csv(self.data_path)
            elif file_extension in ['.xlsx', '.xls']:
                self.data = pd.read_excel(self.data_path)
            else:
                raise ValueError(f"Unsupported file extension: {file_extension}. Please provide a CSV or Excel file.")
            
            print(f"Successfully loaded data with {self.data.shape[0]} rows and {self.data.shape[1]} columns")
            
            # Create output directory if it doesn't exist
            os.makedirs('output', exist_ok=True)
            
            # Save a sample of the data as CSV for easy reference
            sample = self.data.head(10)
            sample.to_csv('output/data_sample.csv', index=False)
            print("Saved data sample to output/data_sample.csv")
            
            return self.data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def get_data_summary(self):
        """
        Get a summary of the dataset
        
        Returns:
            dict: Dictionary containing dataset summary
        """
        if self.data is None:
            print("Data not loaded. Call load_data() first.")
            return None
        
        summary = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'numerical_summary': self.data.describe().to_dict()
        }
        
        # Print summary information
        print("\n=== Dataset Summary ===")
        print(f"Number of rows: {summary['shape'][0]}")
        print(f"Number of columns: {summary['shape'][1]}")
        print("\nColumns:")
        for col in summary['columns']:
            print(f"- {col}")
        
        print("\nMissing values:")
        for col, count in summary['missing_values'].items():
            if count > 0:
                print(f"- {col}: {count} missing values")
        
        # Save summary to text file
        with open('output/data_summary.txt', 'w') as f:
            f.write("=== Dataset Summary ===\n")
            f.write(f"Number of rows: {summary['shape'][0]}\n")
            f.write(f"Number of columns: {summary['shape'][1]}\n\n")
            
            f.write("Columns:\n")
            for col in summary['columns']:
                f.write(f"- {col}\n")
            
            f.write("\nMissing values:\n")
            for col, count in summary['missing_values'].items():
                if count > 0:
                    f.write(f"- {col}: {count} missing values\n")
        
        print("Saved data summary to output/data_summary.txt")
        return summary
    
    def get_data(self):
        """
        Get the loaded dataset
        
        Returns:
            pandas.DataFrame: The loaded dataset
        """
        if self.data is None:
            print("Data not loaded. Call load_data() first.")
        return self.data


# For testing the module independently
if __name__ == "__main__":
    collector = DataCollector("Reports and Dataset/student_habits_dataset.xlsx")
    data = collector.load_data()
    summary = collector.get_data_summary()
    print("\nData collection completed successfully!") 