import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import the main function from src.main_pipeline instead of main
from src.main import main

if __name__ == "__main__":
    main() 