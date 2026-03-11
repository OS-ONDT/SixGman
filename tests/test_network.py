from pathlib import Path

# Get the path to the current script (my_example.py)
current_file = Path(__file__).resolve()

# Go to project root (assumes "examples" is directly under root)
project_root = current_file.parent.parent

# Define path to results directory
src_dir = project_root / "src"
src_dir.mkdir(parents=True, exist_ok=True)  # Ensure it exists

import sys
import os
# Navigate relative to the current working directory
sys.path.append(os.path.abspath(src_dir))
import pytest
from sixgman.core.network import Network

import pytest
from sixgman.core.network import Network

def test_network_initialization():
    """Test that a Network object can be initialized with a sample topology."""
    network = Network(topology_name="MAN157")
    assert network is not None
    assert hasattr(network, "topology_name")
    assert network.topology_name == "MAN157"

