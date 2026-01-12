import numpy as np
import pandas as pd
import networkx as nx
from scipy.io import loadmat
from scipy.sparse.csgraph import yen
from scipy.sparse import csr_matrix
import os
import networkx as nx
from typing import Dict, List, Set, Tuple

class Network:
    """A class representing an optical network topology and its properties.
    
    This class handles the network topology, hierarchical levels, and path computation
    for the SixDman planning tool.

    Attributes:
        graph (nx.Graph): NetworkX graph representing the network topology graph.
        hierarchical_levels (Dict[str, Dict[str, List[str]]]): 
            Dictionary containing nodes organized by hierarchical level.
        topology_name (str): Name of the network topology.
    """

    def __init__(self, 
                 topology_name: str):
        """Initialize a Network instance.
        
        Args:
        ---------
            topology_name (str): 
                Name of the network topology.

        Example
        ---------
        >>> from sixdman.core.network import Network
        >>> net = Network("ExampleNetwork")
        """
        self.graph = nx.Graph()
        self.hierarchical_levels: Dict[str, Dict[str, List[str]]] = {
            'HL1': {'standalone': [], 'colocated': []},
            'HL2': {'standalone': [], 'colocated': []},
            'HL3': {'standalone': [], 'colocated': []},
            'HL4': {'standalone': [], 'colocated': []}
        }
        self.topology_name = topology_name

        
    def load_topology(self, 
                      filepath: str, 
                      matrixName: str = None):
        """Load network topology from .mat, .npz, or .npy file.
        
        This function reads an adjacency matrix from a file, converts it to a 
        NetworkX graph, and initializes related network attributes.
        
        Supported formats:
        ---------
            .mat : 
                MATLAB file (requires `matrixName` to specify the variable)
            .npz : 
                NumPy compressed archive (variable name required)
            .npy : 
                NumPy single-array file

        Args:
        ---------
            filepath (str): 
                Path to the file containing network topology.
            matrixName (str): 
                Name of the adjacency matrix variable in .mat or .npz files.

        Output:
        ---------
            nx.Graph: A NetworkX graph representing the loaded network topology.

        Raises:
        ---------
            FileNotFoundError: 
                If the file does not exist.
            ValueError: 
                If the adjacency matrix is empty or invalid.
            KeyError: 
                If the specified matrixName does not exist in the file.
            IOError: 
                If there is an error loading the file or parsing the matrix.

        Example
        ---------
        >>> net.load_topology("path/to/topology.mat", matrixName="adjacency_matrix")
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Topology file not found: {filepath}")

        ext = os.path.splitext(filepath)[1].lower()
        adjacency_matrix = None

        try:
            if ext == ".mat":
                # Load MATLAB file
                mat_data = loadmat(filepath)
                if matrixName is None:
                    raise ValueError("matrixName is required to load a .mat file.")
                if matrixName not in mat_data:
                    raise KeyError(f"Matrix '{matrixName}' not found in {filepath}")
                adjacency_matrix = mat_data[matrixName]

            elif ext == ".npz":
                # Load compressed NumPy archive
                npz_data = np.load(filepath)
                if matrixName is None:
                    if len(npz_data.files) == 1:
                        matrixName = npz_data.files[0]  # auto-detect single array
                    else:
                        raise ValueError(
                            f"Multiple arrays found in {filepath}. Specify matrixName."
                        )
                if matrixName not in npz_data:
                    raise KeyError(f"Matrix '{matrixName}' not found in {filepath}")
                adjacency_matrix = npz_data[matrixName]

            elif ext == ".npy":
                # Load single NumPy array
                adjacency_matrix = np.load(filepath)
            
            else:
                raise ValueError(f"Unsupported file format: {ext}. Use .mat, .npz, or .npy.")

            # Validate adjacency matrix
            if adjacency_matrix is None or adjacency_matrix.size == 0:
                raise ValueError("The adjacency matrix is empty or invalid.")

            # Store adjacency matrix and convert to upper triangular to avoid duplicate edges
            self.adjacency_matrix = np.triu(adjacency_matrix)

            # Create NetworkX graph
            self.graph = nx.from_numpy_array(self.adjacency_matrix)

            # Extract edges with weights
            self.all_links = np.array(list(self.graph.edges(data="weight")))
            self.weights_array = (
                self.all_links[:, 2] if len(self.all_links) > 0 else np.array([])
            )

            return self.graph

        except Exception as e:
            raise IOError(f"Failed to load topology from {filepath}: {str(e)}")
   
    def define_hierarchy(self, 
                         **kwargs):
        """Set the hierarchical levels of nodes in the network.

        This method allows flexible assignment of nodes to hierarchical levels 
        (HL1, HL2, ...) with optional `standalone` and `colocated` classifications.

        Behavior:
        ---------
            - Accepts keyword arguments like HL1_standalone, HL2_colocated, etc. 
            - `standalone` nodes are unique to that level. 
            - `colocated` nodes are shared with previous levels if not explicitly given. 
            - If a `_colocated` list is not provided, it is **auto-accumulated** from all previous standalone nodes. 
                
        Args:
        ---------
            **kwargs:  
                Variable keyword arguments for HLx_standalone and HLx_colocated. 

        Output:
        ---------
            Dict[str, Dict[str, List[str]]]: 
                Updated hierarchical levels structure. 

        
        Examples
        ---------
        >>> HL_dict = net.define_hierarchy(
        ...     HL3_standalone = [1, 2, 3],
        ...     HL4_standalone = [4, 5, 6, 7],
        ...     HL4_colocated = [1]
        ... )

        >>> HL4_Standalone = hl_dict['HL4']['standalone']
        >>> HL4_colocated = hl_dict['HL4']['colocated']
        >>> HL4_all = np.concatenate((HL4_Standalone, HL4_colocated))
        
        """
        self.hierarchical_levels = {}
        colocated_accum = []  # Keeps track of all standalone nodes for auto-colocation

        # Determine all hierarchical levels (HL1, HL2, ...)
        levels = set()
        for key in kwargs:
            if key.endswith('_standalone'):
                levels.add(key[:-11])  # Remove "_standalone"
            elif key.endswith('_colocated'):
                levels.add(key[:-9])   # Remove "_colocated"
            else:
                # Allow simple HLx key with no suffix
                levels.add(key)

        # Process each hierarchical level in sorted order
        for hl in sorted(levels):
            standalone = kwargs.get(f"{hl}_standalone", [])
            # Auto-fill colocated if user does not provide it
            colocated = kwargs.get(f"{hl}_colocated", colocated_accum.copy())

            # Store in hierarchical_levels
            self.hierarchical_levels[hl] = {
                'standalone': standalone,
                'colocated': sorted(colocated)
            }

            # Update accumulated colocated nodes only if user didn't override
            if f"{hl}_colocated" not in kwargs:
                colocated_accum += standalone

        return self.hierarchical_levels
    
    def _calc_all_hierarchical_nodes(self):
        """Calculate and return all hierarchical nodes across all levels, including both 
            standalone and colocated nodes.

        Behavior:
        ---------
            It uses the internal self.hierarchical_levels attribute to calculate all the nodes with aggregation
            
        Output:
        ---------
            List: 
                list of all hierarchical nodes.

        """
        # Collect all unique nodes from all hierarchical levels in hl_dict
        all_hierarchy_nodes = set()
        for level in self.hierarchical_levels.values():
            for node_type in level.values():
                all_hierarchy_nodes.update(node_type)

        all_hierarchy_nodes = sorted(all_hierarchy_nodes)

        return all_hierarchy_nodes
    
    
    def get_standalone_hierarchy_level(self, node):
        """
        Returns the standalone hierarchical level (HL1, HL2, etc) of the given node.
        If the node is not found in any standalone level, returns None.
        
        Args:
        ---------
            node: 
                The index of the node for which the standalone hierarchical level is to be determined.

        Output:
        ---------
            Integer representing the standalone hierarchical level of the input node (e.g., 2 for HL2).
            
            
        """
        for level_name in ['HL2', 'HL3', 'HL4']:
            if node in self.hierarchical_levels[level_name]['standalone']:
                level_number = level_name.replace('HL', '') if isinstance(level_name, str) else level_name
                return int(level_number)
        return None
    
    def _reconstruct_yen_path(self,
                              predecessors: np.ndarray,
                              path_index: int,
                              source: int,
                              target: int):
        """Reconstruct a path from Yen's algorithm predecessors matrix.

        Args:
        ---------
            predecessors (np.ndarray): 
                Predecessors matrix from Yen's algorithm (shape: [k_paths, n_nodes])
            path_index (int): 
                Index of the path to reconstruct
            source (int): 
                Source node ID
            target (int): 
                Target node ID

        Output:
        ---------
            List[int]: 
                Ordered list of node IDs representing the reconstructed path. Returns an empty list if the path is not reachable.
        """
        path = []
        node = target

        # Traverse predecessor chain backwards until reaching the source
        while node != -9999 and node != source:
            path.append(node)
            node = predecessors[path_index, node]

        if node == -9999:
            return []  # Path not reachable

        path.append(source)
        return path[::-1]  # Reverse to get path from source → target

    
    def _get_link_indices_in_path(self, 
                                 path: List[int]):
        """Return the list of link indices that correspond to the given path.
        
        This function checks both directions (u→v and v→u) because the network graph 
        is undirected.

        Args:
        ---------
            path (List[int]): 
                Ordered list of node IDs representing the path.

        Output:
        ---------
            List[int]: 
                Indices of links in `self.all_links` that form this path.

        Raises:
        ---------
            ValueError: 
                If any consecutive nodes in the path do not correspond to a valid link.
        """
        all_links = self.all_links  # Expected shape: (num_links, 2)
        link_indices = []

        for i in range(len(path) - 1):
            src, dst = path[i], path[i + 1]

            # Try forward direction
            link_idx = np.where((all_links[:, 0] == src) & (all_links[:, 1] == dst))[0]

            # Try reverse direction if not found
            if link_idx.size == 0:
                link_idx = np.where((all_links[:, 0] == dst) & (all_links[:, 1] == src))[0]

            if link_idx.size == 0:
                raise ValueError(f"No link found between nodes {src} and {dst}.")

            link_indices.append(int(link_idx[0]))  # Use first matching link index

        return link_indices
    
    def compute_hierarchy_subgraph(self,
                                   hierarchy_level: int,
                                   minimum_hierarchy_level: int):
        """Extract a subgraph from the network based on hierarchical constraints.

        Constructs a subgraph that includes edges where at least one endpoint is in 
        the specified hierarchy level (HLx), and the other is not in any lower level 
        between HL(x+1) and HL(minimum).

        Args:
        ---------
            hierarchy_level (int): 
                The current HL level (e.g., 1 for HL1).
            minimum_hierarchy_level (int): 
                The lowest HL level to exclude from edge participation.

        Output:
        ---------
            Tuple[nx.Graph, np.ndarray]: 
                - The resulting NetworkX subgraph.
                - The subgraph's adjacency (cost) matrix with np.inf for missing links.

        Example
        ---------    
        >>> subgraph, subnetMatrix = net.compute_hierarchy_subgraph(
        ...     hierarchy_level = 4, # Current hierarchy level
        ...     minimum_hierarchy_level = 4 # Minimum hierarchy level to include in analysis
        ... )
        """

        # Extract nodes for the specified hierarchy level
        try:
            current_nodes = self.hierarchical_levels[f"HL{hierarchy_level}"]['standalone']
        except KeyError:
            raise ValueError(f"HL{hierarchy_level} not found in hierarchical_levels.")

        # Gather all standalone nodes from lower hierarchy levels
        excluded_nodes = []
        for hl in range(hierarchy_level + 1, minimum_hierarchy_level + 1):
            level_key = f"HL{hl}"
            if level_key in self.hierarchical_levels:
                excluded_nodes.extend(self.hierarchical_levels[level_key]['standalone'])

        excluded_nodes = set(excluded_nodes)
        current_nodes = set(current_nodes)

        # Identify edges where one node is in current_nodes and the other is not in excluded_nodes
        edges_in_subgraph = [
            (u, v) for u, v in self.graph.edges
            if (u in current_nodes and v not in excluded_nodes) or
            (v in current_nodes and u not in excluded_nodes)
        ]

        # Initialize an adjacency (cost) matrix for the subgraph
        sub_cost_matrix = np.full_like(self.adjacency_matrix, np.inf, dtype=float)
        for u, v in edges_in_subgraph:
            sub_cost_matrix[u, v] = self.adjacency_matrix[u, v]

        # Build subgraph from the cost matrix
        subgraph = nx.from_numpy_array(
            np.where(np.isfinite(sub_cost_matrix), sub_cost_matrix, 0)
        )

        return subgraph, sub_cost_matrix
    
    def get_neighbor_nodes(self, nodes: List[int]):
        """Return the unique neighbors of a given list of nodes in the graph.

        This excludes the input nodes themselves. Each neighbor appears only once 
        regardless of how many input nodes it is connected to.

        Args:
        ---------
            nodes (List[int]): 
                List of node IDs.

        Output:
        ---------
            List[int]: 
                Sorted list of unique neighbor node IDs.

        Example
        --------    
        >>> neighbors_HL4 = net.get_neighbor_nodes(HL4_standalone)
        """

        # define set to avoid duplicates
        connected_nodes = set() 
        for node in nodes:
            connected_nodes.update(self.graph.neighbors(node))

        # Remove the target nodes themselves from the result
        connected_nodes -= set(nodes)

        return connected_nodes
    
    def compute_k_shortest_paths(self, 
                                 subnet_matrix: np.ndarray,
                                 paths: List[Dict],
                                 source: int,
                                 target: int,
                                 k: int = 20):
        """Compute k-shortest paths between source and target nodes using Yen's algorithm.
        
        This function computes candidate paths for optical network planning, 
        calculates distances, and updates the `paths` list with detailed path information.

        Args:
        ---------
            subnet_matrix (np.ndarray): 
                Adjacency matrix of the subnet.
            paths (List[Dict]): 
                List to append path dictionaries to (can be empty initially).
            source (int): 
                Source node ID.
            target (int): 
                Target node ID.
            k (int, optional): 
                Number of paths to compute (default: 20).

        Output:
        ---------
            List[Dict]: 
                Updated list of paths, where each dictionary contains:
                    - src_node (int): Source node
                    - dest_node (int): Destination node
                    - nodes (List[int]): Sequence of nodes in the path
                    - links (List[int]): Link indices forming the path
                    - distance (float): Total path distance
                    - num_hops (int): Number of hops in the path

        Example
        ---------
        >>> src_nodes = HL4_standalone
        >>> target_nodes = neighbors_HL4
        >>> k_paths = 20

        >>> # Define a list to store path attributes
        >>> K_path_attributes = []
        >>> # Iterate through each standalone HL4 node
        >>> for src in src_nodes:
        ...     for dest in target_nodes:
        ...         K_path_attributes = net.compute_k_shortest_paths(
        ...             subnetMatrix,
        ...             K_path_attributes,
        ...             source = src,
        ...             target = dest,
        ...             k = k_paths
        ...         )
        >>> # Convert to dataframe
        >>> K_path_attributes_df = pd.DataFrame(K_path_attributes)
        """
        # Convert adjacency matrix to sparse format for efficiency
        graph_sparse = csr_matrix(subnet_matrix)
        
        # Compute k-shortest paths using Yen's algorithm
        distances, predecessors = yen(
            csgraph=graph_sparse,
            source=source,
            sink=target,
            K=k,
            directed=False,
            return_predecessors=True
        )

        for i, distance in enumerate(distances):
            if distance == np.inf:
                continue  # Skip unreachable paths

            # Reconstruct path using the updated helper function
            path = self._reconstruct_yen_path(predecessors, i, source, target)
            if not path:
                continue

            # Get link indices for this path
            links = self._get_link_indices_in_path(path)

            # Append structured path information
            paths.append({
                "src_node": int(source),
                "dest_node": int(target),
                "nodes": list(map(int, path)),
                "links": list(map(int, links)),
                "distance": float(distance),
                "num_hops": len(path) - 1
            })

        return paths
    
    def get_node_degrees(self, 
                         nodes: List[int]):
        """Get the degree of specified nodes in the graph.

        This method returns a dictionary where each key is a node ID, and each value
        is the degree of the corresponding node (the number of edges connected to it).

        Args:
        ---------
            nodes (List[int]): 
                List of node IDs for which to retrieve the degree.

        Output:
        ---------
            np.ndarray: 
                2D numpy array containing node IDs and their relative degree.
        """
        return np.array(self.graph.degree(nodes))
            
    def land_pair_finder(self,
                         src_list: List[int],
                         candidate_paths_sorted: pd.DataFrame,
                         num_pairs: int) -> pd.DataFrame:
        """
        Identify link- and node-disjoint path pairs (LAND pairs) for each source node.

        A LAND pair consists of a primary and secondary path where:
        - They connect different destination nodes.
        - They are node-disjoint (except the source).
        - They are link-disjoint.

        Args:
        ---------
            src_list (List[int]): 
                List of source node IDs to process.
            candidate_paths_sorted (pd.DataFrame): 
                DataFrame of candidate paths. Must include columns: ['src_node', 'dest_node', 'nodes', 'links', 'index', 'distance', 'num_hops'].
            num_pairs (int): 
                Maximum number of disjoint pairs to return per source node.

        Output:
        ---------
            pd.DataFrame:  
                DataFrame with columns: ['primary_path_IDx', 'secondary_path_IDx', 'numHops_secondary', 'distance_secondary', 'src_node']

        Example
        ---------
        >>> # Sort the candidate paths by number of hops and distance
        >>> K_path_attributes_df_sorted = K_path_attributes_df.groupby(
        ...     ['src_node'], group_keys = False).apply(
        ...         lambda x: x.sort_values(['num_hops', 'distance'])
        ...     )
        
        >>> # Find 1 disjoint pairs for each source node
        >>> pairs_disjoint = net.land_pair_finder(
        ...     src_nodes, 
        ...     K_path_attributes_df_sorted, 
        ...     num_pairs = 1
        ... )

        """
        results = []

        # Preprocessing: store the index once to avoid repeated use of iterrows
        candidate_paths_sorted = candidate_paths_sorted.reset_index()

        for node in src_list:
            node_df = candidate_paths_sorted[candidate_paths_sorted['src_node'] == node]
            used_secondary_idxs = set()
            pair_counter = 0

            # Precompute all rows as lists for speed
            node_records = node_df.to_dict('records')

            for i, primary in enumerate(node_records):
                if pair_counter >= num_pairs:
                    break

                primary_idx = primary['index']
                dest_primary = primary['dest_node']
                nodes_primary = set(primary['nodes'])
                links_primary = set(primary['links'])

                # Skip primary paths already used as secondary
                if primary_idx in used_secondary_idxs:
                    continue

                # Filter once outside the inner loop
                secondary_candidates = [
                    (j, secondary) for j, secondary in enumerate(node_records)
                    if secondary['dest_node'] != dest_primary and secondary['index'] not in used_secondary_idxs
                ]

                for _, secondary in secondary_candidates:
                    nodes_secondary = set(secondary['nodes'])
                    links_secondary = set(secondary['links'])

                    # Disjoint condition: only source node in common (assumed to be first element)
                    common_nodes = nodes_primary & nodes_secondary
                    common_links = links_primary & links_secondary

                    if len(common_nodes) == 1 and len(common_links) == 0:
                        # Store result
                        results.append([
                            primary_idx,
                            secondary['index'],
                            secondary['num_hops'],
                            secondary['distance'],
                            node
                        ])
                        used_secondary_idxs.add(secondary['index'])
                        pair_counter += 1
                        break  # Only one valid secondary per primary

        # Convert to DataFrame
        standalone_path_df = pd.DataFrame(results, columns=[
            'primary_path_IDx',
            'secondary_path_IDx',
            'numHops_secondary',
            'distance_secondary',
            'src_node'
        ])

        return standalone_path_df
    
    def calc_num_pair(self, 
                      pairs_disjoint_df: pd.DataFrame):

        """calculate the number of candidate link & node disjoint (LAND) pairs.
        
        Args:
        ---------
            pairs_disjoint_df: 
                Dataframe of disjoint_pairs
            
        Output:
        ---------
            A numpy array containing the number of candidate LAND pairs for each source node
        """
        return pairs_disjoint_df.groupby('src_node')['primary_path_IDx'].count().to_numpy()


        