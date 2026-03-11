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
from sixgman.core.band import Band, OpticalParameters
from sixgman.core.network import Network

@pytest.fixture
def c_band_params():
    """Fixture providing typical C-band parameters."""
    return OpticalParameters()

def test_band_initialization(c_band_params):
    network = Network(topology_name="MAN157")
    band = Band(
        name='C',
        start_freq = 190.65, # THz
        end_freq = 196.675, # THz
        opt_params = c_band_params,
        network_instance = network,
        channel_spacing = 0.05 # THz
    )
    
    assert band.name == 'C'
    assert band.num_channels == 121  
    assert len(band.spectrum) == band.num_channels


# Add more tests...