network Module
===============

The ``network`` module provides tools for **modeling optical network topologies**,
loading them from different file formats, and preparing the data for
**planning and simulation** in the SixGman toolkit.

Module Overview
----------------

This module is responsible for:

- Representing an optical network topology using **NetworkX graphs**.
- Loading adjacency matrices from ``.mat``, ``.npz``, or ``.npy`` files.
- Set the hierarchy level of different nodes in the network.
- Compute k-shortest paths between source and destination nodes using Yen's algorithm.
- Identify link- and node-disjoint path pairs (LAND pairs) for each source node. 

Key Class
------------

Network
~~~~~~~~

.. autoclass:: sixgman.core.network.Network
    :members:
    :special-members: __init__
    :undoc-members:
    :show-inheritance: 


Key Methods
------------

- **``__init__(topology_name)``**  
  Initializes the network with a given name and hierarchical structure.

- **``load_topology(filepath, matrixName=None)``**  
  Loads the adjacency matrix from a file and converts it to a NetworkX graph.  
  Supports **.mat**, **.npz**, and **.npy** formats.

- **``define_hierarchy(filepath, matrixName=None)``**  
  Set the hierarchical levels of nodes in the network.

- **``compute_k_shortest_paths(subnet_matrix, paths, source, target, k=20)``**  
  Compute k-shortest paths between source and target nodes using Yen’s algorithm.

- **``land_pair_finder(src_list, candidate_paths_sorted, num_pairs)``**  
  Identify link- and node-disjoint path pairs (LAND pairs) for each source node.
