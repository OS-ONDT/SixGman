band Module
============

This module provides classes for **optical transmission bands** and their **parameters** used in multi-band optical network planning.

Module Overview
----------------

This module is responsible for:

- Defining **optical transmission bands** (e.g., **C-band**, **L-band**) used in multi-band optical network planning.
- Storing and computing **fiber and system parameters** for optical modeling through the ``OpticalParameters`` class.
- Computing **channel frequency grids (spectra)** based on start frequency, end frequency, and channel spacing.
- Associating **band characteristics** with a given network topology.
- Preparing the **frequency plan** for further optical performance evaluation such as GSNR calculations.

Key Classes
------------

- **OpticalParameters**
    ~~~~~~~~~~~~~~~~~~~~
    .. autoclass:: sixdman.core.band.OpticalParameters
        :members:
        :undoc-members:
        :show-inheritance: 

- **Band**
    ~~~~~~~
    .. autoclass:: sixdman.core.band.Band
        :members:
        :special-members: __init__
        :undoc-members:
        :show-inheritance:

Key Methods
------------

- **``process_link_gsnr(f_c_axis, Pch_dBm, num_Ch_mat, spectrum_C, Nspan_array, hierarchy_level, minimum_hierarchy_level, result_directory)``**  
    Processes the GSNR and throughput of all links at a given hierarchy level.  

