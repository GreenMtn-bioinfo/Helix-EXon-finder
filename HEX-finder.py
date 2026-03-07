#!/usr/bin/env python3
import sys
from pathlib import Path

# Add the src directory to the path for convenience
src_path = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(src_path))

# Import the entry point script's main function and run it
from Helix_EXon_Finder.cli import main
if __name__ == "__main__":
    sys.exit(main())