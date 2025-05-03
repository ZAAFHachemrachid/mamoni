import os
import sys

# Add project root and src directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, project_root)
sys.path.insert(0, src_path)

# Import and run main
from src.main import main

if __name__ == "__main__":
    main()