#!/bin/bash

# Function to find the root directory containing the marker file
find_repo_root() {
  local dir="$PWD"
  while [ "$dir" != "/" ]; do
    if [ -f "$dir/.repo_root_marker_hf" ]; then
      echo "$dir"
      return
    fi
    dir=$(dirname "$dir")
  done
  echo ""
}

# Find the repository root
REPO_ROOT=$(find_repo_root)

# Check if the repository root was found
if [ -z "$REPO_ROOT" ]; then
  echo "Please run this script from within the repository directory or a subdirectory."
  exit 1
fi

# Change to the repository root directory
cd "$REPO_ROOT/src/" || exit 1

# Run the main script and pass all arguments
python hex_finder.py "$@"