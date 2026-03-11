post_process Module
===============================

Module Overview
----------------

This module provides utilities for **analyzing and managing results** from 
hierarchical optical network simulations and planning processes.

Its primary responsibilities include:

- Loading **link-level** and **transceiver-level (BVT)** metrics from saved simulation results.
- Managing **results for multiple hierarchy levels** (HL2, HL3, HL4, etc.).
- Providing structured access to **network simulation data** for further processing or visualization.
- Acting as a **post-processing interface** for the SixGman planning framework.

Typical applications include:

- Post-processing optical network planning simulations
- Extracting link utilization, spectral efficiency, and BVT deployment data
- Analyzing results for different **hierarchy levels** in metro/urban networks

Key Class
----------

analyse_result
~~~~~~~~~~~~~~
.. autoclass:: sixgman.core.post_process.analyse_result
    :members:
    :special-members: __init__
    :undoc-members:
    :show-inheritance: 

Key Methods
------------

- **``plot_link_state(splitter, save_flag, save_suffix='', flag_plot=1)``**  
  Plot or return the evolution of link states (Frequency Plan numbers) across all hierarchy levels over time.

- **``plot_FP_usage(save_flag, save_suffix='', flag_plot=1)``**  
  Plot and optionally save the Fiber Pair (FP) usage over time across all hierarchy levels.

- **``plot_FP_degree(save_flag, save_suffix='', flag_plot=1)``**  
  Plot and optionally save the cumulative Fiber Pair (FP) usage and Band degree types.

- **``plot_bvt_license(save_flag, save_suffix='', flag_plot=1)``**  
  Plot and optionally save the cumulative BVT usage and 100G license allocation over time.

- **``calc_cost(save_flag, save_suffix='', C_100GL_First=1, C_100G_Added=0.333, C_MCS=0.7, C_RoB=1.9, C_IRU=0.5)``**  
  Compute OPEX and CAPEX values for network deployment over time.

- **``calc_E2E_latency_Total(latency_core_array, destination_core_array, processing_level_list, save_flag, save_suffix='')``**  
  Compute end-to-end latency across hierarchical processing levels.
