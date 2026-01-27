planning Module
================

This module provides the **planning and optimization core** for the SixDman (6-Dimensional Metro-Area Network) framework.  

Module Overview
----------------

Its primary responsibilities include:

- Integrating **network topology** (nodes, links, hierarchical levels) with **multi-band optical transmission**.
- Performing **traffic-aware and spectrum-aware network planning** for metro and urban transport networks.
- Supporting **multi-band coexistence** by coordinating multiple optical bands (e.g., C, L, S) for transmission.
- Providing a **time-based planning window** for iterative or dynamic network optimization.
- Serving as a **controller** that coordinates modules from:
  
  - ``network.py`` → Topology and hierarchy modeling
  - ``band.py`` → Optical band definition and physical parameters

Key Class
------------

PlanningTool
~~~~~~~~~~~~

.. autoclass:: sixdman.core.planning.PlanningTool
    :members:
    :special-members: __init__
    :undoc-members:
    :show-inheritance: 

Key Methods
------------

- **``run_planner(hierarchy_level, prev_hierarchy_level, pairs_disjoint, kpair_standalone, kpair_colocated, candidate_paths_standalone_df, candidate_paths_colocated_df, GSNR_opt_link, minimum_level, node_cap_update_idx, result_directory)``**  
  Executes the hierarchical planning algorithm for the given hierarchy level.  

