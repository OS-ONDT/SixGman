band Module
===========

This module provides classes for **optical transmission bands** and their
**parameters** used in multi-band optical network planning.

Module Overview
---------------

This module is responsible for:

- Defining **optical transmission bands** (e.g., **C-band**, **L-band**)
- Storing and computing **fiber and system parameters** for optical modeling
- Computing **channel frequency grids (spectra)**
- Associating **band characteristics** with a given network topology
- Preparing the **frequency plan** for optical performance evaluation

Key Classes
-----------

OpticalParameters
~~~~~~~~~~~~~~~~~

.. autoclass:: sixgman.core.band.OpticalParameters
   :members:
   :undoc-members:
   :show-inheritance:

Band
~~~~

.. autoclass:: sixgman.core.band.Band
   :members:
   :special-members: __init__
   :undoc-members:
   :show-inheritance:

Key Methods
-----------

- ``process_link_gsnr(f_c_axis, Pch_dBm, num_Ch_mat, spectrum_C,
  Nspan_array, hierarchy_level, minimum_hierarchy_level, result_directory)``
  
  Processes the GSNR and throughput of all links at a given hierarchy level.
