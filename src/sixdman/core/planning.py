from typing import List, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from .network import Network
from .band import Band
import os

# WSS

@dataclass
class PlanningTool:
    """
    Main class for SixDman (6-Dimensional Metro-Area Network) planning and optimization.
    
    This tool integrates physical layer modeling with hierarchical network topology
    and supports multi-band optical transmission planning. It is intended for
    use in optical metro and urban transport networks where spectral efficiency,
    topology hierarchy, and multi-band coexistence are jointly considered.

    Attributes:
        network (Network): A reference to the network topology object that contains
            all node, link, and hierarchical information used for planning.
        bands (List[Band]): A list of Band instances, each representing an optical band
            (e.g., C, L, S) with its spectral and physical transmission parameters.
        period_time (int): The time period (in arbitrary units) over which traffic 
            demand aggregation or optimization planning is conducted.
    """
    
    def __init__(self,
                 network_instance: Network,
                 bands: List[Band],
                 optical_parameters: dict,
                 grid_center: np.ndarray,
                 period_time: int = 10):
        """
        Initialize the PlanningTool object with a given network topology and bands.

        Args:
            network_instance (Network): The network structure including nodes, links,
                weights, and hierarchical levels.
            bands (List[Band]): The optical bands considered for transmission planning.
                Each Band includes start/end frequencies and physical layer parameters.
            optical_parameters (dict):
                Dictionary containing the optical transmision parameters.
            grid_center (np.ndarray): 
                Numpy array containing the center frequency of frequency channels.
            period_time (int, optional): Time granularity (e.g., 10 units) for periodic
                traffic updates or recalculation windows. Default is 10.

        Example:
        -------
        >>> from sixdman.core.planning import PlanningTool
        
        >>> # Initialize planning tool
        >>> planner = PlanningTool(
        ...     network_instance = net, # the network instance
        ...     bands = [c_band], # list of all band instances
        ...     optical_parameters = c_band_params, # optical transmission parameters
        ...     grid_center = f_c_axis, # center frequncy of frequency channels
        ...     period_time = 10 # the time period for planning (e.g., 10 years)
        ... )

        """
        self.network = network_instance
        self.bands = bands
        self.optical_parameters = optical_parameters
        self.period_time = period_time
        self.grid_center = grid_center
    
    def initialize_planner(self, 
                           num_fslots: int,
                           hierarchy_level: int,
                           minimum_hierarchy_level: int,
                           rolloff: float = 0.1,
                           SR: float = 40 * 1e9,
                           Max_bit_rate_BVT: np.ndarray = np.array([400, 348, 280, 200, 120, 80]),
                           Ref_license_capacity: np.ndarray = np.array([100, 87, 70, 50, 30, 20]), 
                           FP_max_num: int = 20, 
                           band_sepration_idx: list = [96, 120]):
        """
        Initialize planning-related matrices and spectrum parameters.

        This method prepares internal state variables needed for simulating 
        BVT allocations, GSNR calculations, fiber placement tracking, and 
        spectrum usage planning across different optical bands.

        Args:
            num_fslots (int): Number of frequency slots available in the network.
            hierarchy_level (int): Target hierarchy level for current planning.
            minimum_hierarchy_level (int): Minimum hierarchy level for subgraph planning.
            rolloff (float): Rolloff factor for spectral shaping (default: 0.1).
            SR (float): Symbol rate in baud (default: 40 Gbaud).
            Max_bit_rate_BVT (np.ndarray): Array of supported BVT bitrates in Gbps.
            Ref_license_capacity (np.ndarray): Array of reference license capacities of different BVT bitrates.
            FP_max_num (int): Maximum number of available fiber pairs per link.
            band_sepration_idx (list): Index of the last frequency slot of the C-Band and SuperC-Band.

        Example:
        -------
        >>> planner.initialize_planner(
        ...     num_fslots = num_fslots, # number of frequency slots
        ...     hierarchy_level = 4, # current hierarchy level
        ...     minimum_hierarchy_level = 4 # minimum hierarchy levels
        ...     rolloff = 0.1, 
        ...     SR = 64e9, 64 Gbaud symbol rate for 75GHz channel spacing
        ...     Max_bit_rate_BVT = np.array([400, 320, 260, 200, 120, 64]), 
        ...     Ref_license_capacity = np.array([100, 80, 65, 50, 30, 16]),
        ...     FP_max_num = 100 # 100 fiber pairs is available for assignment, 
        ...     band_sepration_idx = [64, 80], # 64 and 80 are the last FS of C-Band and SuperC-Band
        ... )
        """
        # List containing the bitrate of BVTs based on modulation formats
        self.Max_bit_rate_BVT = Max_bit_rate_BVT
        
        # Each BVT contains some licenses, this list containing the capacity of licenses of BVTs based on their bitrate
        self.Ref_license_capacity = Ref_license_capacity

        # Index of the last frequency slot of C-Band
        self.C_band_separation_idx = band_sepration_idx[0]
        
        # Index of the last frequency slot of the SuperC-Band
        # Note that slots of L-Band is from the self.supC_band_separation_idx to the last frequency slot of the whole spectrum.
        self.supC_band_separation_idx = band_sepration_idx[1]
        
        # Get number of HL nodes at given hierarchy level
        num_node_standalone = len(self.network.hierarchical_levels[f"HL{hierarchy_level}"]['standalone'])
        num_node_colocated = len(self.network.hierarchical_levels[f"HL{hierarchy_level}"]['colocated'])
        
        # Get number of links in the whole network
        num_links = len(self.network.all_links)
        
        # Get the whole period of planning horizon
        period_time = self.period_time

        # Generate the planning subgraph of this hierarchical level
        subgraph, _ = self.network.compute_hierarchy_subgraph(hierarchy_level, minimum_hierarchy_level)

        # Assume fixed-grid channel spacing (from first defined band)
        channel_spacing = self.bands[0].channel_spacing
        self.num_fslots = num_fslots

        # Calculate effective channel bandwidth
        B_ch = SR * (1 + rolloff)

        # Calculate how many frequency slots are needed per BVT
        self.Required_FS_BVT = np.ceil(B_ch / (channel_spacing * 1e12)).astype(int)

        # Yearly tracking of fiber placements across all links
        self.Year_FP = np.zeros((period_time, num_links), dtype=np.int32)

        # Track fiber placement for colocated HL nodes specifically
        self.Year_FP_HL_colocated = np.zeros((period_time, num_node_colocated))

        # Residual (unserved) traffic of the last established BVT in standalone HL nodes (primary path - secondary path)
        self.Residual_Throughput_BVT_standalone_HLs_primary = np.zeros((period_time, num_node_standalone))        
        self.Residual_Throughput_BVT_standalone_HLs_secondary = np.zeros((period_time, num_node_standalone))

        # Residual (unserved) traffic of the last established BVT in colocated HL nodes (primary path - secondary path)
        self.Residual_Throughput_BVT_colocated_HLs_primary = np.zeros((period_time, num_node_colocated))
        self.Residual_Throughput_BVT_colocated_HLs_secondary = np.zeros((period_time, num_node_colocated))

        # Total number of BVTs deployed per year (across all bands) in different BVT types (based on modulation format)
        self.HL_BVT_number_all_annual = np.zeros((period_time, len(Max_bit_rate_BVT)))

        # Band-specific BVT deployment tracking
        self.HL_BVT_number_Cband_annual = np.zeros(period_time)         # Traditional C-Band
        self.HL_BVT_number_SuperCband_annual = np.zeros(period_time)      # SuperC-Band
        self.HL_BVT_number_Lband_annual = np.zeros(period_time)     # L-Band
        
        # Track the annual number of BVTs in each node 
        self.HL_BVT_number_per_node = np.zeros(shape = (len(self.Max_bit_rate_BVT), self.period_time, self.network.adjacency_matrix.shape[0]))
                
        
        self.FP_max_num = FP_max_num  # Maximum number of fiber pairs per link
        
        # Optical spectrum tracking (Link State Profile for standalone and colocated paths)
        self.LSP_array = np.zeros((self.num_fslots, num_links, self.FP_max_num))
        self.LSP_array_Colocated = np.zeros((self.num_fslots, num_node_colocated, self.FP_max_num)) 

        # Band usage tracking (link-level statistics per year)
        self.num_link_CBand_annual = np.zeros(shape = (period_time, num_links), dtype = np.int32)
        self.num_link_SupCBand_annual = np.zeros(shape = (period_time, num_links), dtype = np.int32)
        self.num_link_LBand_annual = np.zeros(shape = (period_time, num_links), dtype = np.int32)

        # Fiber placement for new subgraph connections
        self.Year_FP_new = np.zeros((period_time, subgraph.number_of_edges()))

        # Effective fiber placement deployment tracking
        self.Total_effective_FP_new_annual = np.zeros(period_time)
        self.Total_effective_FP = np.zeros(period_time)

        # Capacity profile (aggregated traffic) for each node across years and hierarchy levels
        self.node_capacity_profile_array = np.zeros(
            shape=(period_time, self.network.adjacency_matrix.shape[0], minimum_hierarchy_level)
        )

        # Annua license usage tracking per node
        self.num_100G_licence_annual = np.zeros(shape=(period_time, self.network.adjacency_matrix.shape[0])) # Track total licenses that activated in each year
        self.num_100G_licence_CBand_annual = np.zeros(shape=(period_time, self.network.adjacency_matrix.shape[0])) # Track licenses that activated in C-Band
        self.num_100G_licence_superCBand_annual = np.zeros(shape=(period_time, self.network.adjacency_matrix.shape[0])) # Track licenses that activated in SuperC-Band
        self.num_100G_licence_LBand_annual = np.zeros(shape=(period_time, self.network.adjacency_matrix.shape[0])) # Track licenses that activated in L-Band
        
        # Total traffic flow per link per year
        self.traffic_flow_links_array = np.zeros((self.period_time, num_links), dtype = float)
    
        # for each node, a list that contain the info of BVTs, the info of each BVT is like this [FSs used, FP used, links used, primary/secondary, established BVT bitrate, established BVT GSNR]
        # Note: in this array primary path is specified by 1 and secondary path is specified by -1 
        self.BVT_establishment_info = [[] for _ in range(self.network.adjacency_matrix.shape[0])] # BVT establishment information storage

        # Trach the residual capacity of the last activated license in each node in each year
        self.Residual_Throughput_LC_standalone_HLs_primary = np.zeros((period_time, num_node_standalone)) # BVTs for primary path of standalone nodes
        self.Residual_Throughput_LC_standalone_HLs_secondary = np.zeros((period_time, num_node_standalone)) # BVTs for secondary path of standalone nodes
        self.Residual_Throughput_LC_colocated_HLs_primary = np.zeros((period_time, num_node_colocated)) # BVTs for primary path of colocated nodes
        self.Residual_Throughput_LC_colocated_HLs_secondary = np.zeros((period_time, num_node_colocated)) # BVTs for primary path of colocated nodes
        
        # Track the number of activated licenses in the last BVT of each node
        self.num_license_last_BVT_primary = np.zeros(self.network.adjacency_matrix.shape[0]) # primary path
        self.num_license_last_BVT_secondary = np.zeros(self.network.adjacency_matrix.shape[0]) # secondary path
        
        # Track the Type of last established BVT in each node (bitrate of BVT based on modulation format)
        self.last_BVT_type_primary = np.zeros(self.network.adjacency_matrix.shape[0]) # primary path
        self.last_BVT_type_secondary = np.zeros(self.network.adjacency_matrix.shape[0]) # secondary path
        
        # Track the Band in which the last BVT of each node is established
        self.last_BVT_Band_primary = np.zeros(self.network.adjacency_matrix.shape[0]) # primary path
        self.last_BVT_Band_secondary = np.zeros(self.network.adjacency_matrix.shape[0]) # secondary path
        
        # Save the GSNR of BVTs in each year
        self.GSNR_BVT_primary_annual = np.zeros(self.period_time, dtype = object) # BVTs of primary paths
        self.GSNR_BVT_secondary_annual = np.zeros(self.period_time, dtype = object) # BVTs of secondary paths
        self.GSNR_BVT_all_annual = np.zeros(self.period_time, dtype = object) # BVTs of all paths
        
        # Store links of the LAND pairs for each node
        self.LAND_Links_Storage = np.zeros(shape = self.network.adjacency_matrix.shape[0], dtype=object)
        
        all_nodes = self.network._calc_all_hierarchical_nodes()
        # Track the path latency of each standalone path (5 microsecond/km)
        self.path_latency_storage = np.zeros(len(all_nodes), dtype = object)
        # Track the destination of each standalone path
        self.destinations_storage = np.zeros(len(all_nodes), dtype = object)
    
    def generate_initial_traffic_profile(self,
                                        num_nodes: int,
                                        optical_nodes: list,
                                        monteCarlo_steps: int,
                                        min_rate: float,
                                        max_rate: float,
                                        seed: int, 
                                        result_directory) -> np.ndarray:
        """
        Generate or load the initial traffic capacity profile for network nodes using Monte Carlo simulation.

        This method estimates the initial traffic demand or capacity for each network node by 
        performing multiple Monte Carlo simulations. For each iteration, it generates random 
        capacities uniformly distributed between the specified minimum and maximum rates. 
        If a precomputed capacity file exists in the given directory, it is loaded instead 
        to avoid redundant computation.

        The resulting per-node average capacities are stored internally in 
        `self.HL_capacity_final`. This method does not return any value.

        Args:
        ---------
            num_nodes (int): 
                Number of nodes in the network for which to simulate traffic.
            optical_nodes (list): 
                List of all nodes with no electrical aggregation (optically bypass nodes), these nodes not have any initial traffic and can't be the destination of any node
            monteCarlo_steps (int): 
                Number of Monte Carlo iterations used for averaging.
            min_rate (float): 
                Minimum possible traffic rate (Gbps) per node.
            max_rate (float): 
                Maximum possible traffic rate (Gbps) per node.
            seed (int): 
                Initial random seed for reproducibility of the simulations.
            result_directory (Path): 
                Directory where results are stored or loaded from.

        Updates:
        ---------
            self.HL_capacity_final (np.ndarray): 
                Final per-node traffic capacity values averaged over all Monte Carlo simulations.

        Output:
        ---------
            None

        Example:
        ---------
        >>> # generate port capacity for the lowest HL nodes uisng Monte Carlo simulation
        >>> planner.generate_initial_traffic_profile(
        ...     num_nodes = len(HL4_all), # all the nodes of minimum hierarchy level
        ...     Optical_nodes = [], # there is no any all optical node
        ...     monteCarlo_steps = 100, # Number of Monte Carlo iterations
        ...     min_rate = 20, # minimum allowed traffic rate per node in Gbps
        ...     max_rate = 200, # maximum allowed traffic rate per node in Gbps
        ...     seed = 20, # random seed for reproducibility
        ...     result_directory = results_dir # Path to the directory where results are stored
        ... )

        """
        
        # Define the filename for storing/loading precomputed capacity results
        file_path = result_directory / f"{self.network.topology_name}_HL_capacity_final.npz"

        # Load precomputed initial traffic if exists
        if os.path.exists(file_path):
            
            print("Loading precomputed HL_capacity_final ...")
            data = np.load(file_path)
            self.HL_capacity_final = data['HL_capacity_final']

        else:
            print("Calculate HL_capacity_final ...")

            # Storage for traffic capacity samples across Monte Carlo runs
            random_capacity_storage = []

            for i in range(monteCarlo_steps):
                # Set seed each iteration (to ensure consistent output if seed is constant)
                np.random.seed(seed + i)

                # Generate uniform random capacity for each node
                random_capacity_local = np.random.uniform(min_rate, max_rate, size=num_nodes)

                # Store this realization
                random_capacity_storage.append(random_capacity_local)
            
            # Convert the random_capacity_storage list to numpy array
            random_capacity_storage = np.array(random_capacity_storage)

            # Average traffic capacities across all Monte Carlo simulations
            self.HL_capacity_final = random_capacity_storage.mean(axis=0)
            
            # Set the initial traffic of optical nodes to zero
            if len(optical_nodes) != 0:
                self.HL_capacity_final[optical_nodes] = 0

            # Save computed capacity to disk
            np.savez_compressed(file_path, HL_capacity_final=self.HL_capacity_final)

    def simulate_traffic_annual(self,
                                 lowest_hierarchy_dict: dict,
                                 CAGR: int, 
                                 result_directory) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate annual traffic evolution for the lowest hierarchy-level nodes using a Compound Annual Growth Rate (CAGR).

        This method models how network traffic grows over multiple years at the lowest hierarchy
        level (e.g., HL4) by applying a constant annual growth rate to each node’s base capacity.
        It calculates annual traffic, added traffic, number of 100G licenses, and residual capacities
        for both standalone and colocated nodes.

        If precomputed results are available in the specified directory, they are loaded from file
        to save computation time. Otherwise, the full annual simulation is performed and results are saved.

        Args:
        ---------
            lowest_hierarchy_dict (dict): 
                Dictionary containing node IDs for the lowest hierarchy level, with keys `'standalone'` and `'colocated'`.
            CAGR (float): 
                Compound Annual Growth Rate (e.g., 0.4 for 40% annual increase).
            result_directory (Path): 
                Directory where results are read from or written to.

        Updates:
        ---------
            self.lowest_HL_added_traffic_annual_standalone (np.ndarray): 
                Annual incremental traffic for standalone nodes (years × nodes).
            self.lowest_HL_added_traffic_annual_colocated (np.ndarray): 
                Annual incremental traffic for colocated nodes (years × nodes).
        
        Output:
        ---------
            None

        Example:
        ---------
        >>> # Traffic growth simulation over 10 years
        >>> planner.simulate_traffic_annual(
        ...     lowest_hierarchy_dict = hl_dict['HL4'], # Dictionary with minimum hierarchy level standalone and colocated nodes
        ...     CAGR = 0.4, # 40% annual growth rate
        ...     result_directory = results_dir # Path to the directory where results are stored
        ...)
        """
        
        # Get node counts
        nodes_standalone = lowest_hierarchy_dict['standalone']
        nodes_colocated = lowest_hierarchy_dict['colocated']
        num_node_total = self.network.adjacency_matrix.shape[0]
        period_time = self.period_time

        # Path for cached results
        file_path = result_directory / f"{self.network.topology_name}_traffic_matrix.npz"

        if os.path.exists(file_path):
            # Load precomputed traffic data
            print("Loading precomputed Traffic Matrix ...")
            data = np.load(file_path)
            added_traffic_annual = data['added_traffic_annual']

            # Split added traffic into standalone and colocated components
            self.lowest_HL_added_traffic_annual_standalone = added_traffic_annual[:, nodes_standalone]
            self.lowest_HL_added_traffic_annual_colocated = added_traffic_annual[:, nodes_colocated]

        else:
            print("Calculate Traffic Matrix ...")

            # Preallocate data structures for annual metrics
            lowest_HL_traffic_storage_annual = np.empty((period_time, num_node_total))
            total_traffic_annual = np.empty(period_time)
            added_traffic_annual = np.empty((period_time, num_node_total))

            # Initialize first year with current capacity
            lowest_HL_traffic_storage_annual[0, :] = self.HL_capacity_final
            total_traffic_annual[0] = np.sum(self.HL_capacity_final)
            added_traffic_annual[0, :] = self.HL_capacity_final

            # Iterate over years and apply CAGR growth
            for year in range(1, period_time):
                
                # Apply CAGR to simulate traffic growth
                lowest_HL_traffic_storage_annual[year, :] = (
                    (1 + CAGR) * lowest_HL_traffic_storage_annual[year - 1, :]
                )

                # Compute total network traffic for the year
                total_traffic_annual[year] = np.sum(lowest_HL_traffic_storage_annual[year, :])

                # Compute incremental traffic compared to previous year
                added_traffic_annual[year, :] = (
                    lowest_HL_traffic_storage_annual[year, :] - lowest_HL_traffic_storage_annual[year - 1, :]
                )

            # Final separation of standalone and colocated traffic data
            self.lowest_HL_added_traffic_annual_standalone = added_traffic_annual[:, nodes_standalone]
            self.lowest_HL_added_traffic_annual_colocated = added_traffic_annual[:, nodes_colocated]

            # Persist computed data to file
            np.savez_compressed(file_path, added_traffic_annual = added_traffic_annual)             

    def _spectrum_assignment(self,
                            path_IDx: int,
                            path_type: str,
                            year: int, 
                            K_path_attributes_df: pd.DataFrame,
                            pure_traffic_to_assign: float,
                            BVT_number: int,
                            node_IDx: int,
                            node_list: List,
                            GSNR_link: np.ndarray, 
                            LSP_array_pair: np.ndarray, 
                            Year_FP_pair: np.ndarray, 
                            HL_subnet_links = np.ndarray) -> dict:
        """
        Perform spectrum and fiber pair assignment for a given lightpath in a hierarchical network.

        This function implements a first-fit spectrum assignment algorithm to allocate frequency 
        slots (FS) and fiber pairs for a path. It supports primary and secondary paths for 
        standalone HL nodes, as well as co-located HL nodes. It calculates per-BVT costs, 
        tracks spectrum occupancy, and updates GSNR for each assigned path.

        Args:
        --------
            path_IDx (int or None): 
                Index of the selected path in K_path_attributes_df. If None, the function performs assignment for a colocated node.
            path_type (str): 
                Type of path ('primary' or 'secondary').
            year (int): 
                Planning year for multi-period simulation.
            K_path_attributes_df (pd.DataFrame): 
                DataFrame containing attributes of all K-shortest paths.
            pure_traffic_to_assign (float):
                Pure traffic value of this node that must be routed in this year 
            BVT_number (int): 
                Number of BVTs to assign for this path.
            node_IDx (int): 
                Index of the current node being processed.
            node_list (list): 
                List of all node identifiers.
            GSNR_link (np.ndarray): 
                GSNR values per frequency slot per link.
            LSP_array_pair (np.ndarray): 
                Spectrum occupancy array [FS, link, fiber pair].
            Year_FP_pair (np.ndarray): 
                Annual fiber pair usage array [year, link].
            HL_subnet_links (np.ndarray): 
                List of high-level subnetwork link indices.

        Updates:
        --------
            LSP_array_pair: 
                Updates spectrum occupancy for allocated frequency slots and fiber pairs.
            Year_FP_pair: 
                Updates annual fiber pair usage for links in the assigned path.
            Year_FP_HL_colocated: 
                Updated for colocated HL nodes when path_IDx is None.
            self.num_added_license_this_year_primary / _secondary: 
                Stores number of added licenses.
            BVT_CBand_count_path, BVT_superCBand_count_path, BVT_superCLBand_count_path:
                Counts BVTs assigned in each spectral band.
            self.HL_BVT_number_all_annual (np.ndarray): 
                Annual count of deployed BVTs across all nodes.
            self.HL_BVT_number_per_node (np.ndarray): 
                Annual count of deployed BVTs per node.
            self.BVT_establishment_info (list):
                Information of established BVTs per node.
            self.num_license_last_BVT_primary (np.ndarray):
                Number of activated licenses in the last established BVT (primary path).
            self.num_license_last_BVT_secondary (np.ndarray):
                Number of activated licenses in the last established BVT (secondary path).
            self.last_BVT_type_primary (np.ndarray):
                Bitrate of the last established BVT (primary) per node. 
            self.last_BVT_type_secondary (np.ndarray):
                Bitrate of the last established BVT (secondary) per node. 
            self.last_BVT_Band_primary (np.ndarray):
                Band in which the last BVT (primary) is established. (1: C-Band, 2: SuperC-Band, 3: L-Band)
            self.last_BVT_Band_secondary (np.ndarray):
                Band in which the last BVT (secondary) is established. (1: C-Band, 2: SuperC-Band, 3: L-Band)
                
        Output:
        --------
            dict, np.ndarray, np.ndarray
            If path_IDx is not None (normal path):
                - path_info_storage (dict): Contains fiber usage, BVT counts per band, 
                    information and allocated frequency slots of established BVTs, cost metrics, and information of path like links, distance, number of hops and destination node.
                - LSP_array_pair (np.ndarray): Updated spectrum occupancy.
                - Year_FP_pair (np.ndarray): Updated fiber usage per link.
            If path_IDx is None (colocated primary paths):
                - Year_FP_HL_colocated (np.ndarray): Updated fiber pair usage for colocated HL node.
                - BVT_bitrate_storage (np.ndarray): Store the bitrate of established BVTs
                - FS_BVT_storage (list): Store the allocated frequency slots of established BVTs
        
        Notes:
        --------
            - Uses first-fit strategy: tries to find contiguous free slots exactly matching BVT requirement, otherwise picks the first larger available block.
            - Assigns FS in C-band, SuperC-band, and SuperCL-band.
            - Stops searching for further slots once a valid assignment is made.
            - Calculate the total bitrate of established BVTs and continue BVT allocation to reach the pure_traffic_to_assign of this node.
        """
        if path_IDx != None:
            
            # Initialize dict for path information
            path_info_storage = {}
            
            # Determine how many frequency slots (FS) are required for the selected BVT type 
            BVT_required_FS_HL = self.Required_FS_BVT
            
            # Initialize counters for BVT allocations in different spectrum bands
            BVT_CBand_count_path = 0
            BVT_supCBand_count_path  = 0
            BVT_LBand_count_path  = 0

            # Extract the link list for the primary path from K_path_attributes_df
            linkList_path = np.array(K_path_attributes_df.iloc[path_IDx]['links'])

            # Extract the number of hops for the primary path from K_path_attributes_df
            numHops_path = K_path_attributes_df.iloc[path_IDx]['num_hops']

            # Extract the destination node of the path
            destination_path = int(K_path_attributes_df.iloc[path_IDx]['dest_node'])
            
            # Extract nodes of this path
            nodes_path = np.array(K_path_attributes_df.iloc[path_IDx]['nodes'])
            
            # Store some path information in dictionary
            path_info_storage['distance'] = K_path_attributes_df.iloc[path_IDx]['distance']
            path_info_storage['links'] = linkList_path
            path_info_storage['numHops'] = numHops_path

            # Initialize FP_counter_links with ones, representing the first available fiber pair for each link 
            FP_counter_links = np.ones(len(linkList_path), dtype = np.int8)

            # Store congested links in the primary path
            link_congested_path = np.array(
                        [np.count_nonzero(LSP_array_pair[:, link, FP_counter_links[i] - 1]) for i, link in
                         enumerate(linkList_path)])
            
            # Sort the unique congestion levels in descending order
            unique_sorted_link_congested_primary = np.sort(np.unique(link_congested_path))[::-1]

            # Sort links based on congestion values
            linkList_path_sorted = np.concatenate(
                [linkList_path[link_congested_path == congestion] for congestion in
                    unique_sorted_link_congested_primary])

            ###################################################
            #  Fiber and spectrum assignment for path
            ###################################################

            fiber_counter = 0 # Define a counter for fiber pair
            f_max_path = [] # Initialize an array for Maximum frequency slot used
            cost_FP_all_BVT_path = [] # Initialize an array for Cost function values
            FP_max_path  = [] # Initialize an array for Maximum fiber pairs assigned
            FS_BVT_storage = [] # Store allocated frequency slots of each established BVT
            BVT_info_local = [] # Store the information of each established BVT
            GSNR_BVT_Storage = [] # Store the GSNR of each established BVT
            BVT_bitrate_storage = [] # Store the bitrate of each established BVT
            
            flag_complete_allocation = 0 # Define a flag that show the completness of BVT allocation in this node
            BVT_counter = 0 # Initialize the BVT_counter 

            while flag_complete_allocation == 0 and BVT_counter < BVT_number:
                      
                Flag_SA_continue_path = 1 # spectrum assignment continues until an available frequency slot is found

                while Flag_SA_continue_path:

                    PST_path = np.zeros(self.num_fslots) # PST_parimary is a binary vector that will store whether each Frequency Slot is occupied or available

                    for FS in range(self.num_fslots): # iterate through each frequency slot
                        
                        vector_state_FS = np.empty(len(linkList_path_sorted))  # vector_state_FS will contain one value per link, indicating whether the slot is free (0) or used (1) on a certain link

                        for link_idx in range(len(linkList_path_sorted)): # check the status of the current frequency slot (FS) for each link

                            vector_state_FS[link_idx] = LSP_array_pair[FS, linkList_path_sorted[link_idx], FP_counter_links[link_idx] - 1] # LSP_array_pair contain a non-zero number for each allocated FS in each link 

                        if any(vector_state_FS): # check that there is any link that use that frequecy slot or not
                            PST_path[FS] = 1
                        else:
                            PST_path[FS] = 0

                    
                    FS_count = 0 # keep track of the number of contiguous free slots
                    PST_vector_aux = np.diff(np.concatenate(([1], PST_path, [1])), n = 1) # PST_vector_aux stores differences in spectrum occupancy
                    flag_First_Fit = 1 # this flag ensures that if exact-fit slots aren’t found, the first available larger slot is chosen
                    FS_path = [] # stores the selected frequency slots
                    
                    if np.any(PST_vector_aux):

                        startIndex = np.where(PST_vector_aux < 0)[0] # find the first index that 0 changes to 1 (start of free block)                   
                        endIndex = np.where(PST_vector_aux > 0)[0] - 1 # find the first index that 1 changes to 0 (end of free block)
                        duration = endIndex - startIndex + 1 # compute the length of each contiguous free block

                        Exact_Fit = np.where(duration == BVT_required_FS_HL)[0] # search for exactly matching free blocks
                        First_Fit = np.where(duration > BVT_required_FS_HL)[0] # search for the first block that match

                        if Exact_Fit.size > 0: # if Exact_Fit is found select the first exact-fit slot and assigns it
                            
                            FS_count = duration[Exact_Fit[0]] # select the first available exact-fit slot
                            b_1 = np.arange(startIndex[Exact_Fit[0]], startIndex[Exact_Fit[0]] + BVT_required_FS_HL)
                            FS_path = b_1[:BVT_required_FS_HL]
                            flag_First_Fit = 0
    
                        elif First_Fit.size > 0 and flag_First_Fit: # if no Exact-Fit, use First-Fit
                            
                            FS_count = duration[First_Fit[0]] # select the first available larger slot
                            b_1 = np.arange(startIndex[First_Fit[0]],
                                            startIndex[First_Fit[0]] + BVT_required_FS_HL)
                            FS_path = b_1[:BVT_required_FS_HL]
    
                    if FS_count >= BVT_required_FS_HL: # if enough contiguous slots are found, the assignment proceeds

                        GSNR_BVT1 = [0]
                        flag_GSNR_calculation = 0
                        for link_idx in range(len(linkList_path_sorted)):
                            
                            if path_type == 'primary':
                                LSP_array_pair[FS_path, linkList_path_sorted[link_idx], FP_counter_links[link_idx] - 1] = (node_list[node_IDx] + 1) # update LSP_array_pair to reflect the new assignment
                            elif path_type == 'secondary':
                                LSP_array_pair[FS_path, linkList_path_sorted[link_idx], FP_counter_links[link_idx] - 1] = -(node_list[node_IDx] + 1) # update LSP_array_pair to reflect the new assignment with a negative identifier
                                
                            link_in_subnet = np.where(HL_subnet_links == linkList_path_sorted[link_idx])[0]
                            if len(link_in_subnet) != 0:
                                flag_GSNR_calculation = 1

                                for n_span in range(int(self.Nspan_array[linkList_path_sorted[link_idx]])):
                                    GSNR_BVT1 += (10 ** (GSNR_link[link_in_subnet, FS_path] / 10)) ** -1 # compute GSNR

                        GSNR_connection = 10 * np.log10((GSNR_BVT1[0] + 10 ** -3.6) ** -1)
                        GSNR_connection = GSNR_connection - 1 - self.wss_penalty_degree[self.all_node_degree[nodes_path[link_idx]]]
                        
                        # Check for the GSNR threshold
                        mod_index = np.searchsorted(-self.optical_parameters.target_SNR_dB, -GSNR_connection, side='left')
                        
                        if mod_index != 0:
                            print('GSNR below threshold of 64-QAM')
                        
                        BVT_bitrate_storage.append(self.Max_bit_rate_BVT[mod_index])
                        Flag_SA_continue_path = 0 # stop searching for more slots
                        
                        for link_counter_local in range(len(FP_counter_links)):

                            Year_FP_pair[year - 1, linkList_path_sorted[link_counter_local]] =  max(Year_FP_pair[year - 1, linkList_path_sorted[link_counter_local]], FP_counter_links[link_counter_local]) # update the Year_FP_pair to record spectrum usage for each link

                        cost_FP_all_BVT_path.append(np.dot(FP_counter_links, self.network.weights_array[linkList_path_sorted])) # calculate the cost of assigning fiber pairs for the BVT_counter-th BVT

                        # Store the last allocated frequency slot of the established BVT 
                        FS_BVT_storage.append(FS_path[-1].copy())
                        
                        # Track the BVT numbers per band based on the last allocated frequency slot
                        if FS_path[-1] < self.C_band_separation_idx: 
                            BVT_CBand_count_path += 2 # the coeficient 2 is due to the same BVT in the destination
                        elif self.C_band_separation_idx <= FS_path[-1] < self.supC_band_separation_idx:
                            BVT_supCBand_count_path += 2 # the coeficient 2 is due to the same BVT in the destination
                        else:
                            BVT_LBand_count_path += 2 # the coeficient 2 is due to the same BVT in the destination

                    else: # If no suitable spectrum was found, move to the next FP link

                        fiber_counter = (fiber_counter + 1) % len(FP_counter_links)
                        FP_counter_links[fiber_counter - 1] += 1

                f_max_path.append(max(FS_path)) # store the highest frequency slot index used for this BVT
                FP_max_path.append(max(FP_counter_links)) # record the maximum fiber pair used
                
                
                if path_type == 'primary': 
                    
                    # Store GSNR of established BVT
                    GSNR_BVT_Storage.append(GSNR_connection)
                    
                    # Reorder the FP_counter_links in the main LinkList array
                    value_to_calc = dict(zip(linkList_path_sorted, FP_counter_links))
                    FP_counter_links_main = np.array([value_to_calc[x] for x in linkList_path])

                    # Save the information of established BVT
                    BVT_info_local.append([FS_path, 
                                           FP_counter_links_main.copy() - 1, 
                                           linkList_path, 
                                           1, # 1 for primary path
                                           self.Max_bit_rate_BVT[mod_index], 
                                           GSNR_connection]) # store the frequency slots and fiber pairs used for this BVT
                
                elif path_type == 'secondary':
                    if flag_GSNR_calculation != 0:
                        
                        # Store GSNR of established BVT
                        GSNR_BVT_Storage.append(GSNR_connection)
                    
                    # Reorder the FP_counter_links in the main LinkList array
                    value_to_calc = dict(zip(linkList_path_sorted, FP_counter_links))
                    FP_counter_links_main = np.array([value_to_calc[x] for x in linkList_path])
                    
                    # Save the information of established BVT
                    BVT_info_local.append([FS_path, 
                                           FP_counter_links_main.copy() - 1, 
                                           linkList_path,  
                                           -1, # -1 for secondary path
                                           self.Max_bit_rate_BVT[mod_index], 
                                           GSNR_connection]) # store the frequency slots and fiber pairs used for this BVT
                
                # check for the final BVT required   
                if BVT_counter == BVT_number - 1:
                    total_traffic_BVTs = np.sum(BVT_bitrate_storage)
                    
                    # If the total bitrate of established BVTs is less than the pure_traffic_to_assign, try to assign another BVT
                    if total_traffic_BVTs < pure_traffic_to_assign:
                        BVT_number += 1
                    else:
                        flag_complete_allocation = 1

                BVT_counter += 1
                        
            if path_type == 'primary':
                
                # Calculate number of licenses in the full established BVTs
                self.num_added_license_this_year_primary += 4 * (len(BVT_bitrate_storage) - 1) # each Full BVT requires 4 licenses of the corresponding capacity
                
                # Find the type of last established BVT
                last_BVT_type_index = np.where(self.Max_bit_rate_BVT == BVT_bitrate_storage[-1])[0] 
                
                # Calculate number of licenses in the last established BVT
                num_license_last_BVT = np.ceil((pure_traffic_to_assign - np.sum(BVT_bitrate_storage[:-1])) / self.Ref_license_capacity[last_BVT_type_index])
                
                # Update number of licenses 
                self.num_added_license_this_year_primary += num_license_last_BVT
                
                # Store number of added licenses of this path
                path_info_storage['num_added_license'] = self.num_added_license_this_year_primary
                
                # Store the number of activated licenses in the last established BVT
                self.num_license_last_BVT_primary[node_list[node_IDx]] = num_license_last_BVT
                
                # Store type of the last established BVT (based on bitrate)
                self.last_BVT_type_primary[node_list[node_IDx]] = BVT_bitrate_storage[-1]
                
                # Store the band in which the last established BVT is assigned
                if FS_BVT_storage[-1] < self.C_band_separation_idx:
                    self.last_BVT_Band_primary[node_list[node_IDx]] = 1 # C-Band
                elif self.C_band_separation_idx <= FS_BVT_storage[-1] < self.supC_band_separation_idx:
                    self.last_BVT_Band_primary[node_list[node_IDx]] = 2 # SuperC-Band
                else:
                    self.last_BVT_Band_primary[node_list[node_IDx]] = 3 # L-Band
                
            elif path_type == 'secondary':
                
                # Calculate number of licenses in the full established BVTs
                self.num_added_license_this_year_secondary += 4 * (len(BVT_bitrate_storage) - 1) # each Full BVT requires 4 licenses of the corresponding capacity
                
                # Find the type of last established BVT
                last_BVT_type_index = np.where(self.Max_bit_rate_BVT == BVT_bitrate_storage[-1])[0]   
                
                # Calculate number of licenses in the last established BVT
                num_license_last_BVT = np.ceil((pure_traffic_to_assign - np.sum(BVT_bitrate_storage[:-1])) / self.Ref_license_capacity[last_BVT_type_index])
                
                # Update number of licenses 
                self.num_added_license_this_year_secondary += num_license_last_BVT
                
                # Store number of added licenses of this path
                path_info_storage['num_added_license'] = self.num_added_license_this_year_secondary
                
                # Store the number of activated licenses in the last established BVT
                self.num_license_last_BVT_secondary[node_list[node_IDx]] = num_license_last_BVT
                
                # Store type of the last established BVT (based on bitrate)
                self.last_BVT_type_secondary[node_list[node_IDx]] = BVT_bitrate_storage[-1]
                
                # Store the band in which the last established BVT is assigned
                if FS_BVT_storage[-1] < self.C_band_separation_idx:
                    self.last_BVT_Band_secondary[node_list[node_IDx]] = 1
                elif self.C_band_separation_idx <= FS_BVT_storage[-1] < self.supC_band_separation_idx:
                    self.last_BVT_Band_secondary[node_list[node_IDx]] = 2
                else:
                    self.last_BVT_Band_secondary[node_list[node_IDx]] = 3
            
            # Update the BVT info of this node and add newly established BVTs 
            BVT_Info = self.BVT_establishment_info[node_list[node_IDx]]
            BVT_Info.extend(BVT_info_local)
            self.BVT_establishment_info[node_list[node_IDx]] = BVT_Info
            
            # Calculate the total number of BVTs based on bitrate and Update the annual and per-node BVT number
            uniq, counts = np.unique(BVT_bitrate_storage, return_counts=True)
            count_dict = dict(zip(uniq, counts))
            ref_counts = np.array([count_dict.get(x, 0) for x in self.Max_bit_rate_BVT])
            self.HL_BVT_number_per_node[:, year -1, node_list[node_IDx]] += 2 * ref_counts # the coeficient 2 is due to the same BVT in the destination
            self.HL_BVT_number_all_annual[year - 1, :] += 2 * ref_counts # the coeficient 2 is due to the same BVT in the destination
            
            # Update path information dictionary
            path_info_storage['FS_BVT_storage'] = FS_BVT_storage
            path_info_storage['cost_FP'] = cost_FP_all_BVT_path
            path_info_storage['GSNR_Storage'] = GSNR_BVT_Storage
            path_info_storage['f_max'] = f_max_path
            path_info_storage['FP_max'] = FP_max_path
            path_info_storage['BVT_CBand_count'] = BVT_CBand_count_path
            path_info_storage['BVT_superCBand_count'] = BVT_supCBand_count_path
            path_info_storage['BVT_LBand_count'] = BVT_LBand_count_path
            path_info_storage['destination'] = destination_path
            path_info_storage['BVT_Info'] = BVT_info_local
            path_info_storage['BVT_bitrate_storage'] = BVT_bitrate_storage

            return path_info_storage, LSP_array_pair, Year_FP_pair
        
        else: # if path_IDX is None (primary colocated paths)

            FP_counter_links = 0  # define a counter for fibers
            BVT_required_FS_HL = self.Required_FS_BVT # determine how many frequency slots (FS) are required for the selected BVT type 
            
            FS_BVT_storage = []

            for BVT_counter in range(BVT_number): # iterate through BVTs
                
                Flag_SA_continue_path = 1 # spectrum assignment continues until an available frequency slot is found

                while Flag_SA_continue_path:

                    PST_path = self.LSP_array_Colocated[:, node_IDx, FP_counter_links].T # PST_path is a binary vector that will store whether each Frequency Slot is occupied or available
                    FS_count = 0 # keep track of the number of contiguous free slots
                    PST_vector_aux = np.diff(np.concatenate(([1], PST_path, [1])), n = 1) # PST_vector_aux stores differences in spectrum occupancy

                    flag_First_Fit = 1 # this flag ensures that if exact-fit slots aren’t found, the first available larger slot is chosen 
                    FS_path = [] # stores the selected frequency slots
                    
                    if np.any(PST_vector_aux != 0):

                        startIndex = np.where(PST_vector_aux < 0)[0] # find the first index that 0 changes to 1 (start of free block)
                        endIndex = np.where(PST_vector_aux > 0)[0] - 1 # find the first index that 1 changes to 0 (end of free block)     
                        duration = endIndex - startIndex + 1 # compute the length of each contiguous free block

                        Exact_Fit = np.where(duration == BVT_required_FS_HL)[0] # search for exactly matching free blocks
                        First_Fit = np.where(duration > BVT_required_FS_HL)[0] # search for the first block that match

                        if Exact_Fit.size > 0: # if Exact_Fit is found select the first exact-fit slot and assigns it

                            FS_count = duration[Exact_Fit[0]]
                            b_1 = np.arange(startIndex[Exact_Fit[0]], endIndex[Exact_Fit[0]] + 1)
                            FS_path = b_1[:BVT_required_FS_HL]
                            flag_First_Fit = 0

                        elif First_Fit.size > 0 and flag_First_Fit: # if no Exact-Fit, use First-Fit

                            FS_count = duration[First_Fit[0]]
                            b_1 = np.arange(startIndex[First_Fit[0]],
                                            startIndex[First_Fit[0]] + BVT_required_FS_HL)
                            FS_path = b_1[:BVT_required_FS_HL]
                                     
                    if FS_count >= BVT_required_FS_HL: # if enough contiguous slots are found, the assignment proceeds

                        self.LSP_array_Colocated[FS_path, node_IDx, FP_counter_links] = 1 # update LSP_array_pair to reflect the new assignment
                        Flag_SA_continue_path = 0 # stop searching for more slots
                        self.Year_FP_HL_colocated[year - 1, node_IDx] =  max(self.Year_FP_HL_colocated[year - 1, node_IDx], FP_counter_links + 1) # update the Year_FP_pair to record spectrum usage for each link
                       
                        # Store the last frequncy slot of the established BVT
                        FS_BVT_storage.append(FS_path[-1].copy())
                       
                        # Track the BVT numbers per band based on the last allocated frequency slot
                        if FS_path[-1] < self.C_band_separation_idx:
                            self.HL_BVT_number_Cband_annual[year - 1] += 2 # the coeficient 2 is due to the same BVT in the destination
                        elif self.C_band_separation_idx <= FS_path[-1] < self.supC_band_separation_idx: # The final frequency slot used (FS_primary(end)) determines the spectrum band
                            self.HL_BVT_number_SuperCband_annual[year - 1] += 2 # the coeficient 2 is due to the same BVT in the destination
                        else:
                            self.HL_BVT_number_Lband_annual[year - 1] += 2 # the coeficient 2 is due to the same BVT in the destination
                    
                    
                    else: # If no slots are available, increment the fiber pair counter and retry
                        FP_counter_links = FP_counter_links + 1
            
            # Always establish the BVT with highest bitrate for primary colocated path
            BVT_bitrate_storage = self.Max_bit_rate_BVT[0] * np.ones(BVT_number)
            
            # Calculate number of licenses in the full established BVTs
            self.num_added_license_this_year_primary += 4 * (len(BVT_bitrate_storage) - 1) # each Full BVT requires 4 licenses of the corresponding capacity
            
            # Find the type of last established BVT
            last_BVT_type_index = np.where(self.Max_bit_rate_BVT == BVT_bitrate_storage[-1])[0] 
            
            # Calculate number of licenses in the last established BVT
            num_license_last_BVT = np.ceil((pure_traffic_to_assign - np.sum(BVT_bitrate_storage[:-1])) / self.Ref_license_capacity[last_BVT_type_index])
            
            # Update number of licenses 
            self.num_added_license_this_year_primary += num_license_last_BVT
            
            # Store the number of activated licenses in the last established BVT
            self.num_license_last_BVT_primary[node_list[node_IDx]] = num_license_last_BVT
            
            # Store type of the last established BVT (based on bitrate)
            self.last_BVT_type_primary[node_list[node_IDx]] = BVT_bitrate_storage[-1]
            
            # Store the band in which the last established BVT is assigned
            if FS_BVT_storage[-1] < self.C_band_separation_idx:
                self.last_BVT_Band_primary[node_list[node_IDx]] = 1
            elif self.C_band_separation_idx <= FS_BVT_storage[-1] < self.supC_band_separation_idx:
                self.last_BVT_Band_primary[node_list[node_IDx]] = 2
            else:
                self.last_BVT_Band_primary[node_list[node_IDx]] = 3
                
            uniq, counts = np.unique(BVT_bitrate_storage, return_counts=True)
            count_dict = dict(zip(uniq, counts))
            ref_counts = np.array([count_dict.get(x, 0) for x in self.Max_bit_rate_BVT])
            
            self.HL_BVT_number_per_node[:, year -1, node_list[node_IDx]] += 2 * ref_counts
            
            # update BVT allocation tracking, multiplying by 2
            self.HL_BVT_number_all_annual[year - 1, :] += 2 * ref_counts
                
            return self.Year_FP_HL_colocated, BVT_bitrate_storage, FS_BVT_storage
        
    def _update_hl_node_degrees(self, 
                                hierarchy_level: dict,
                                Year_FP: np.ndarray) -> np.ndarray:
        """
        Update and track the average node degree of nodes in the {hierarchy_level} across the planning period.

        This method computes how the average degree (number of active fiber-pair connections) 
        of HL nodes evolves over time based on annual fiber-pair allocations (`Year_FP`). 
        It compares each year's fiber-pair matrix with the previous year to determine 
        new or removed connections and updates node degrees accordingly.

        The resulting yearly average node degrees are both stored internally and returned.

        Args:
        --------
            hierarchy_level (int): 
                The current HL hierarchy level to analyze (e.g., 4 for HL4 nodes).
            Year_FP (np.ndarray): 
                A 2D array of shape (years × links) indicating the number of allocated fiber pairs per link for each year.

        Updates:
        --------
            self.degree_number_HLs (np.ndarray): 
                Average node degree of HL nodes for each simulated year.
            
        Output:
        --------
            None

        """
        
        HL_Standalone = self.network.hierarchical_levels[f"HL{hierarchy_level}"]['standalone'] # Extract standalone HL nodes
        HL_degrees = self.network.get_node_degrees(HL_Standalone) # Get initial node degrees for HL nodes (degree per node)
        degree_node_all_topo_HL_final = HL_degrees.copy() # Create a copy to track node degrees evolution across years

        degree_number_HLs = np.zeros(self.period_time) # Initialize array to store average node degree for each year
        degree_number_HLs[0] = np.mean(HL_degrees[:, 1]) # Record initial average node degree (baseline for year 1)

        # Loop through each year (starting from year 2, since year 1 is baseline)
        for year in range(2, self.period_time + 1):

            # Iterate through each link in the network
            for link_counter in range(len(self.network.all_links)):

                # Check if the fiber pair allocation changed between this year and the previous year
                if Year_FP[year - 1, link_counter] != Year_FP[year - 2, link_counter]:
                    src_node = self.network.all_links[link_counter, 0]
                    dest_node = self.network.all_links[link_counter, 1]

                    # Update degree for source node if it is an HL node
                    if src_node in HL_Standalone:
                        indices = np.where(HL_Standalone == src_node)[0]
                        degree_node_all_topo_HL_final[indices, 1] += (
                            Year_FP[year - 1, link_counter] - Year_FP[year - 2, link_counter]
                        )

                    # Update degree for destination node if it is an HL node
                    if dest_node in HL_Standalone:
                        indices = np.where(HL_Standalone == dest_node)[0]
                        degree_node_all_topo_HL_final[indices, 1] += (
                            Year_FP[year - 1, link_counter] - Year_FP[year - 2, link_counter]
                        )

            degree_number_HLs[year - 1] = np.mean(degree_node_all_topo_HL_final[:, 1]) # Compute average HL node degree for the current year
 
        self.degree_number_HLs = degree_number_HLs

    def _calculate_BVT_usage(self) -> dict:
        """
        Calculate cumulative BVT (Bitrate Variable Transceiver) usage and 100G license counts per year.

        This method computes the cumulative number of deployed BVTs across all optical bands 
        (C, SuperC, and L) and the total 100G license usage for each year in the planning period.
        The results are aggregated annually and stored as instance attributes for later reporting 
        or visualization.

        The calculation assumes four 100G licenses per BVT unit and accumulates counts 
        across all years to reflect cumulative infrastructure growth.

        Updates:
        ---------
            self.HL_All_100G_lincense (np.ndarray):
                Cumulative total 100G license usage across all nodes for each year.
            self.HL_BVTNum_All (np.ndarray):
                Cumulative total number of BVTs (all bands combined) per year.
            self.HL_BVTNum_CBand (np.ndarray):
                Cumulative number of C-band BVTs per year.
            self.HL_BVTNum_SuperCBand (np.ndarray):
                Cumulative number of Super C-band BVTs per year.
            self.HL_BVTNum_LBand (np.ndarray):
                Cumulative number of L-band BVTs per year.

        Output:
        ---------
            None
            
        """
        self.HL_BVTNum_All = self.HL_BVT_number_all_annual.cumsum(axis = 0)
        self.HL_BVTNum_LBand = self.HL_BVT_number_Lband_annual.cumsum(axis = 0)
        self.HL_BVTNum_CBand = self.HL_BVT_number_Cband_annual.cumsum(axis = 0)
        self.HL_BVTNum_SuperCBand = self.HL_BVT_number_SuperCband_annual.cumsum(axis = 0)
        self.HL_All_100G_lincense = (self.num_100G_licence_annual.cumsum(axis = 0) * 2).sum(axis = 1) # the coeficient 2 is due to the same license in the destination

    def _save_network_results(self,
                         hierarchy_level: int,
                         minimum_hierarchy_level: int,
                         result_directory):
        """
        Save detailed network planning results for a given hierarchy level to compressed NPZ files.

        This function generates the subgraph corresponding to the current hierarchy level,
        extracts relevant hierarchical-level (HL) link indices, computes degree metrics per link and 
        spectral band, and saves various results including BVT allocations, link usage, 
        node capacity profiles, traffic flows, and GSNR measurements.

        Args:
        ---------
            hierarchy_level (int): The current hierarchy level being analyzed.
            minimum_hierarchy_level (int): The minimum hierarchy level considered for subgraph generation.
            result_directory (Path): Directory where the output files will be saved.

        Saves:
        ---------

            1. `{topology_name}_HL{hierarchy_level}_bvt_info.npz`:
                - `HL_All_100G_lincense`: Cumulative number of activated 100G licenses.
                - `HL_annual_license`: Number of activated 100G licenses per year.
                - `HL_CBand_license`: Number of activated 100G licenses per year in C-Band.
                - `HL_SuperCBand_license`: Number of activated 100G licenses per year in SuperC-Band.
                - `HL_LBand_license`: Number of activated 100G licenses per year in L-Band.
                - `HL_BVTNum_All`: Cumulative number of deployed BVTs.
                - `HL_BVTNum_CBand`: Cumulative number of deployed BVTs in the C-Band.
                - `HL_BVTNum_SuperCBand`: Cumulative number of deployed BVTs in the SuperC-Band.
                - `HL_BVTNum_LBand`: Cumulative number of deployed BVTs in the L-Band.
                - `BVT_establishment_info`: Information of established BVTs.

            2. `{topology_name}_HL{hierarchy_level}_link_info.npz`:
                - `HL_links_indices`: Indices of HL links within the network graph.
                - `num_link_CBand_annual`: Cumulative number of FPs that used in C-Band in each link.
                - `num_link_SupCBand_annual`: Cumulative number of FPs that used in SuperC-Band in each link.
                - `num_link_LBand_annual`: Cumulative number of FPs that used in L-Band in each link.
                - `HL_CDegree_Domain`: Weighted degree for C-Band links.
                - `HL_SuperCDegree_Domain`: Weighted degree for SuperC-Band links.
                - `HL_LDegree_Domain`: Weighted degree for L-Band links.
                - `Total_effective_FP_new_annual`: total km of fiber pair usage across all links per year.
                - `HL_FPNum`: Fiber pair usage per link per year.
                - `HL_FPNumCo`: Fiber pair usage per colocated HL node per year.
                - `degree_number_HLs`: Node degree per HL node.
                - `traffic_flow_links_array`: Added traffic flow on each link per year.

            3. `{topology_name}_HL{hierarchy_level}_path_GSNR_info.npz`:
                - `GSNR_all_paths`: GSNR values for all BVTs per year.
                - `GSNR_primary`: GSNR values for BVTs of primary paths per year.
                - `GSNR_secondary`: GSNR values for BVTs of secondary paths per year.

            4. `{topology_name}_HL{hierarchy_level}_node_capacity_profile_array.npz`:
                - `node_capacity_profile_array`: Node capacity evolution per year, including allocations and residual capacities.
            
            5. `{topology_name}_HL{hierarchy_level}_segments_latency.npz`:
                - `latency` (np.ndarray): Array where each element corresponds to a node and contains a tuple 
                    with the latency (in microseconds) of the primary and secondary paths from that node.
                - `destinations` (np.ndarray): Array where each element corresponds to a node and contains a tuple 
                    with the destination node indices of the primary and secondary paths from that node.

        Notes:
        ---------
            - The NPZ files are compressed for efficient storage.
            - Arrays typically have shape `[years x links]` or `[years x nodes]` depending on the metric.
            - Frequency band usage arrays (C, Super C, L) allow post-analysis of spectrum utilization.
            - Traffic flow and GSNR arrays allow performance evaluation of deployed paths.
        """   

        # Generate the subgraph for the given hierarchy level
        subgraph, _ = self.network.compute_hierarchy_subgraph(hierarchy_level, minimum_hierarchy_level)

        # Extract HL link indices (filter from all links)
        HL_subnet_links = np.array(list(subgraph.edges(data='weight')))
        mask = np.any(np.all(self.network.all_links[:, None] == HL_subnet_links, axis=2), axis=1)
        HL_links_indices = np.where(mask)[0]

        # Calculate degree per domain (each link adds 2 degrees: one per endpoint)
        HL_CDegree_Domain = 2 * self.num_link_CBand_annual
        HL_SuperCDegree_Domain = 2 * self.num_link_SupCBand_annual
        HL_LDegree_Domain = 2 * self.num_link_LBand_annual

        # Save BVT-related information
        np.savez_compressed(result_directory / f'{self.network.topology_name}_HL{hierarchy_level}_bvt_info.npz',
                            HL_All_100G_lincense=self.HL_All_100G_lincense,
                            HL_annual_license = self.num_100G_licence_annual * 2, # the coefficient 2 is for same license in the destination
                            HL_CBand_license = self.num_100G_licence_CBand_annual * 2,
                            HL_SuperCBand_license = self.num_100G_licence_superCBand_annual * 2,
                            HL_LBand_license = self.num_100G_licence_LBand_annual * 2, 
                            HL_BVTNum_All = self.HL_BVTNum_All,
                            HL_BVTNum_CBand = self.HL_BVTNum_CBand,
                            HL_BVTNum_SuperCBand = self.HL_BVTNum_SuperCBand,
                            HL_BVTNum_LBand = self.HL_BVTNum_LBand, 
                            BVT_establishment_info = np.array(self.BVT_establishment_info, dtype=object))

        # Save link-related information and usage statistics
        np.savez_compressed(result_directory / f'{self.network.topology_name}_HL{hierarchy_level}_link_info.npz',
                            HL_links_indices = HL_links_indices,
                            num_link_CBand_annual = self.num_link_CBand_annual,
                            num_link_SupCBand_annual = self.num_link_SupCBand_annual,
                            num_link_LBand_annual = self.num_link_LBand_annual,
                            HL_CDegree_Domain = HL_CDegree_Domain,
                            HL_SuperCDegree_Domain = HL_SuperCDegree_Domain,
                            HL_LDegree_Domain = HL_LDegree_Domain,
                            Total_effective_FP_new_annual = self.Total_effective_FP_new_annual,
                            HL_FPNum = self.Year_FP_new,
                            HL_FPNumCo = self.Year_FP_HL_colocated,
                            degree_number_HLs = self.degree_number_HLs,
                            traffic_flow_links_array = self.traffic_flow_links_array)
        
        # Save GSNR informations
        np.savez_compressed(result_directory / f'{self.network.topology_name}_HL{hierarchy_level}_path_GSNR_info.npz',
                            GSNR_all_paths = self.GSNR_BVT_all_annual,
                            GSNR_primary = self.GSNR_BVT_primary_annual,
                            GSNR_secondary = self.GSNR_BVT_secondary_annual)

        # Save node capacity profile
        np.savez_compressed(result_directory / f'{self.network.topology_name}_HL{hierarchy_level}_node_capacity_profile_array.npz',
                            node_capacity_profile_array = self.node_capacity_profile_array)
        
        # Save segments latency
        np.savez_compressed(result_directory / f"{self.network.topology_name}_HL{hierarchy_level}_segments_latency.npz", 
                            latency = self.path_latency_storage, 
                            destinations = self.destinations_storage)
        
    def run_planner(self, 
                    hierarchy_level: int,
                    prev_hierarchy_level: int,
                    pairs_disjoint: pd.DataFrame,
                    kpair_standalone: int,
                    kpair_colocated: int,
                    candidate_paths_standalone_df: pd.DataFrame,
                    candidate_paths_colocated_df: pd.DataFrame,
                    GSNR_opt_link: np.ndarray,
                    Nspan_array: np.ndarray, 
                    all_node_degree: np.ndarray,
                    P_opt_links: np.ndarray, 
                    minimum_level: int,
                    node_cap_update_idx: int, 
                    result_directory) -> float:
        """
        Executes the hierarchical optical network planning algorithm for a given hierarchy level.
    
        This function performs traffic allocation, spectrum assignment, and resource planning
        for both standalone and colocated High-Level (HL) nodes across multiple years. 
        It computes primary and secondary paths, assigns BVTs (Bandwidth Variable Transceivers),
        updates frequency plans (FPs), tracks GSNR (Generalized Signal-to-Noise Ratio) evolution,
        and saves annual network performance results.

        The planner operates iteratively per year and per HL node, handling:
            - Traffic growth and residual throughput updates
            - Spectrum assignment via `_spectrum_assignment()`
            - BVT count and license tracking
            - Frequency Plan (FP) and link utilization updates
            - GSNR computation and aggregation over simulation years
            - Capacity profile updates for source and destination nodes

        Args:
        ---------
            hierarchy_level (int): 
                The current hierarchy level being processed.
            prev_hierarchy_level (int): 
                The previous hierarchy level used for continuity and reference.
            pairs_disjoint (pd.DataFrame): 
                DataFrame containing disjoint node pairs for routing and path computation.
            kpair_standalone (int): 
                Maximum number of LAND pairs to consider for standalone HL nodes.
            kpair_colocated (int): 
                Maximum number of K-shortest paths to consider for colocated HL nodes (secondary path).
            candidate_paths_standalone_df (pd.DataFrame): 
                DataFrame of candidate paths for standalone HL pairs, including metrics like hops.
            candidate_paths_colocated_df (pd.DataFrame): 
                DataFrame of candidate paths for colocated HL pairs, used for colocated spectrum assignment.
            GSNR_opt_link (np.ndarray): 
                Array of per-link GSNR (Generalized Signal-to-Noise Ratio) values.
            Nspan_array (np.ndarray): 
                Array that contains number of spans per link in the whole network.
            all_node_degree (np.ndarray): 
                Initial node degress for WSS penalty calculation.
            minimum_level (int): 
                Minimum hierarchy level considered in the network for FP continuity and reference.
            node_cap_update_idx (int): 
                Index in the node capacity array that determines where new capacity values are stored.
            result_directory (Path or str): 
                Directory path where annual and summary results are saved.

        Updates:
        ---------
            
            self.Residual_Throughput_BVT_standalone_HLs_primary (np.ndarray): 
                Residual unallocated throughput for primary paths of standalone nodes.
            self.Residual_Throughput_BVT_standalone_HLs_secondary (np.ndarray): 
                Residual unallocated throughput for secondary paths of standalone nodes.
            self.Residual_Throughput_BVT_colocated_HLs_primary (np.ndarray): 
                Residual unallocated throughput for primary paths of colocated nodes.
            self.Residual_Throughput_BVT_colocated_HLs_secondary (np.ndarray): 
                Residual unallocated throughput for secondary paths of colocated nodes.
            self.Year_FP (np.ndarray): 
                Number of fiber pairs in different years for network links.
            self.Year_FP_HL_colocated (np.ndarray): 
                Number of fiber pairs in years in-site links.
            self.Year_FP_new (np.ndarray): 
                Number of fiber pairs in different years for network links based on spectrum assignment.
            self.Total_effective_FP_new_annual (np.ndarray): 
                Total km of fiber pair usage across all links per year.
            self.GSNR_BVT_all_annual (np.ndarray): 
                GSNR records of all BVTs (primary and secondary) per year.
            self.GSNR_BVT_primary_annual (np.ndarray): 
                GSNR records of primary BVTs per year.
            self.GSNR_BVT_secondary_annual (np.ndarray): 
                GSNR records of secondary BVTs per year.
            self.node_capacity_profile_array (np.ndarray): 
                Node capacity evolution per year, including allocations and residual capacities.
            self.traffic_flow_links_array (np.ndarray): 
                Annual traffic volume per network link.
            self.num_100G_licence_annual (np.ndarray): 
                Annual count of activated 100G licenses.
            self.num_100G_licence_CBand_annual (np.ndarray): 
                Annual count of activated 100G licenses in C-Band.    
            self.num_100G_licence_superCBand_annual (np.ndarray): 
                Annual count of activated 100G licenses in SuperC-Band.  
            self.num_100G_licence_LBand_annual (np.ndarray): 
                Annual count of activated 100G licenses in L-Band.  
            self.Residual_Throughput_LC_standalone_HLs_primary (np.ndarray): 
                Residual capacity of the last activated licenses (primary path) per standalone node.
            self.Residual_Throughput_LC_standalone_HLs_secondary (np.ndarray): 
                Residual capacity of the last activated licenses (secondary path) per standalone node.
            self.Residual_Throughput_LC_colocated_HLs_primary (np.ndarray): 
                Residual capacity of the last activated licenses (primary path) per colocated node.
            self.Residual_Throughput_LC_colocated_HLs_secondary (np.ndarray): 
                Residual capacity of the last activated licenses (secondary path) per colocated node.
            self.LAND_Links_Storage (np.ndarray): 
                Links of selected LAND pair for each node.
            self.LSP_array (np.ndarray): 
                Link-State-Profile array that show the occupied frequency slots in different fiber pairs.
            self.LSP_array_Colocated (np.ndarray):
                Link-State-Profile array that show the occupied frequency slots in different fiber pairs for primary colocated paths.
            self.path_latency_storage (list): 
                Latency records for primary and secondary paths.
            self.destinations_storage (list): 
                Destination node records for primary and secondary paths.
            self.num_link_CBand_annual (np.ndarray):
                Number of fiber pairs with at least one allocated C-Band FS in each link per year.
            self.num_link_SupCBand_annual (np.ndarray):
                Number of fiber pairs with at least one allocated SuperC-Band FS in each link per year.
            self.num_link_LBand_annual (np.ndarray):
                 Number of fiber pairs with at least one allocated L-Band FS in each link per year.
        Output:
        ---------
            None

        Example:
        ---------
        >>> planner.run_planner(hierarchy_level = 4, # Current hierarchy level
        ...         prev_hierarchy_level = 3, # Previous hierarchy level
        ...         pairs_disjoint = pairs_disjoint, # DataFrame of disjoint LAND pairs
        ...         kpair_standalone = 1, # Maximum Number of K-shortest paths for standalone HL nodes
        ...         kpair_colocated = 1, # Maximum Number of K-shortest paths for colocated HL nodes
        ...         candidate_paths_standalone_df = K_path_attributes_df, # DataFrame of candidate paths for standalone HL nodes
        ...         candidate_paths_colocated_df = K_path_attributes_colocated_df, # DataFrame of candidate paths for colocated HL nodes
        ...         GSNR_opt_link = GSNR_opt_link, # GSNR values for each link in this hierarchy level
        ...         Nspan_array = np.ones(len(all_links)), # number of spans for each link in the whole network
        ...         P_opt_links = opt_power_links, # Optimum power values for each link
        ...         minimum_level = 4, # Minimum hierarchy level
        ...         node_cap_update_idx = 2, # Index of node capacity vector to update
        ...         result_directory = results_dir # Directory to save results
        ... )
        """
        self.Nspan_array = Nspan_array
        self.wss_penalty_degree = [0, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 3.5, 4, 5, 6, 7, 8]
        self.all_node_degree = all_node_degree
        self.candidate_paths_standalone_df = candidate_paths_standalone_df
        self.candidate_paths_colocated_df = candidate_paths_colocated_df
        self.pairs_disjoint = pairs_disjoint
        self.minimum_level = minimum_level
        self.P_opt_links = P_opt_links
        period_time = self.period_time
        
        # Extract the standalone and colocated nodes in current hierarchical level
        HL_standalone = self.network.hierarchical_levels[f"HL{hierarchy_level}"]['standalone']
        HL_colocated = self.network.hierarchical_levels[f"HL{hierarchy_level}"]['colocated']

        # Calculate SubNetwork link indices
        subgraph, _ = self.network.compute_hierarchy_subgraph(hierarchy_level, minimum_level)
        HL_subnet_links = np.array(list(subgraph.edges(data = 'weight')))
        mask = np.any(np.all(self.network.all_links[:, None] == HL_subnet_links, axis=2), axis=1)
        HL_links_indices = np.where(mask)[0]

        # array for saving destinations of standalone nodes in each year, in the third dimension 0 is for primary destination and 1 is for secondary destination
        HL_standalone_dest_profile = np.zeros(shape = (period_time, len(HL_standalone), 2), dtype = np.int32)

        # array for saving destinations of colocated nodes in each year, in the third dimension 0 is for primary destination and 1 is for secondary destination
        HL_colocated_dest_profile = np.zeros(shape = (period_time, len(HL_colocated)), dtype = np.int32)
        
        for year in range(1 , period_time + 1):
            
            print('Processing Year: ', year)

            if hierarchy_level == minimum_level:
                    # Create node_capacity_profile array in the minimum hierarchy level
                    node_capacity_profile = np.zeros(shape = (self.network.adjacency_matrix.shape[0], minimum_level))
            else:
                    # Load node_capacity_profile array of previous hierarchy level
                    node_capacity_profile_array_prev_hl = np.load(result_directory /  f"{self.network.topology_name}_HL{prev_hierarchy_level}_node_capacity_profile_array.npz")['node_capacity_profile_array']
                    node_capacity_profile = node_capacity_profile_array_prev_hl[year - 1, :, :]

                    # Load band-degree tracking, traffic-flow of links and latency of previous hierarchical level (just in the first year)
                    if year == 1:
                        
                        # Load Band usage from previous hierarchy level
                        self.num_link_CBand_annual = np.load(result_directory /  f'{self.network.topology_name}_HL{prev_hierarchy_level}_link_info.npz')['num_link_CBand_annual']
                        self.num_link_SupCBand_annual = np.load(result_directory /  f'{self.network.topology_name}_HL{prev_hierarchy_level}_link_info.npz')['num_link_SupCBand_annual']
                        self.num_link_LBand_annual = np.load(result_directory /  f'{self.network.topology_name}_HL{prev_hierarchy_level}_link_info.npz')['num_link_LBand_annual']
                        
                        # Load traffic_flow_links_array from previous hierarchy level
                        self.traffic_flow_links_array = np.load(result_directory /  f'{self.network.topology_name}_HL{prev_hierarchy_level}_link_info.npz')['traffic_flow_links_array']
                        
                        # Load path latency and destinations from previous hierarchy level
                        self.path_latency_storage = np.load(result_directory / f"{self.network.topology_name}_HL{prev_hierarchy_level}_segments_latency.npz", allow_pickle = True)['latency']
                        self.destinations_storage = np.load(result_directory / f"{self.network.topology_name}_HL{prev_hierarchy_level}_segments_latency.npz", allow_pickle = True)['destinations']
                        

            #######################################################
            # Part 1: Spectrum assignment for standalone HL nodes
            #######################################################
            
            for node_idx in range(len(HL_standalone)): # Iterate through each standalone node
                
                # Initialize residual and added licenses (primary and secondary paths) of this node
                num_res_license_prev_year_primary = 0
                self.num_added_license_this_year_primary = 0
                num_res_license_prev_year_secondary = 0
                self.num_added_license_this_year_secondary = 0
                
                print(f"Processing standalone node {HL_standalone[node_idx]}")
                                
                # get traffic demand for this node in this year
                if hierarchy_level == minimum_level:
                    HL_needed_traffic = self.lowest_HL_added_traffic_annual_standalone[year - 1, node_idx]
                else:
                    HL_needed_traffic = node_capacity_profile[HL_standalone[node_idx], node_cap_update_idx + 1]
                
                
                if year != 1: # if it isnt the first year          
                    
                    # subtract residual throughput (unallocated traffic from previous years)         
                    HL_pure_throughput_to_assign_primary = HL_needed_traffic - self.Residual_Throughput_BVT_standalone_HLs_primary[year - 2, node_idx] # primary path
                    HL_pure_throughput_to_assign_secondary = HL_needed_traffic - self.Residual_Throughput_BVT_standalone_HLs_secondary[year - 2, node_idx] # secondary path
                    
                else: # if it is the first year
                    HL_pure_throughput_to_assign_primary = HL_needed_traffic
                    HL_pure_throughput_to_assign_secondary = HL_needed_traffic
                    
                if hierarchy_level == minimum_level:
                    
                    # store traffic capacity assigned to current node
                    node_capacity_profile[HL_standalone[node_idx], node_cap_update_idx + 1] = HL_needed_traffic # store traffic capacity assigned to current node
                
                if year != 1:
                    
                    # calculate the number of not activated licenses of the last established BVT from prevoous year
                    Residual_cap_primary_prev_year = np.round(self.Residual_Throughput_BVT_standalone_HLs_primary[year - 2, node_idx] - self.Residual_Throughput_LC_standalone_HLs_primary[year - 2, node_idx])
                    Residual_cap_secondary_prev_year = np.round(self.Residual_Throughput_BVT_standalone_HLs_secondary[year - 2, node_idx] - self.Residual_Throughput_LC_standalone_HLs_secondary[year - 2, node_idx])
                    index_primary = np.where(self.Max_bit_rate_BVT == self.last_BVT_type_primary[HL_standalone[node_idx]])[0] # type of the last BVT of primary path
                    index_secondary = np.where(self.Max_bit_rate_BVT == self.last_BVT_type_secondary[HL_standalone[node_idx]])[0] # type of the last BVT of secondary path
                    num_res_license_prev_year_primary = int(Residual_cap_primary_prev_year / self.Ref_license_capacity[index_primary])
                    res_license_prev_year_primary_band = self.last_BVT_Band_primary[HL_standalone[node_idx]] # Specify the band in which the last established BVT is assigned (primary)
                    num_res_license_prev_year_secondary = int(Residual_cap_secondary_prev_year / self.Ref_license_capacity[index_secondary])
                    res_license_prev_year_secondary_band = self.last_BVT_Band_secondary[HL_standalone[node_idx]] # Specify the band in which the last established BVT is assigned (secondary)
                                    
                
                if HL_pure_throughput_to_assign_primary > 0 and HL_pure_throughput_to_assign_secondary <= 0:
                    print('one path need to be assigned')
                elif HL_pure_throughput_to_assign_primary <= 0 and HL_pure_throughput_to_assign_secondary > 0:
                    print('one path need to be assigned')
                
                # Extract the first precomputed K-shortest paths for the current standalone node 
                candidate_path_pair = pairs_disjoint[pairs_disjoint['src_node'] == HL_standalone[node_idx]]

                # Calculate the number of LAND pairs for this standalone node
                num_K_pair_final = self.network.calc_num_pair(pairs_disjoint_df = pairs_disjoint)
                num_kpairs = min(num_K_pair_final[node_idx], kpair_standalone) # the num_kpairs for this standalone node is minimum of available k_pair and minimum allowed pairs

                # Initialize the cost function matrix with infinity values for each metric (f_max, N_hop, cost, GSNR, FP_max)
                cost_func = np.full((num_kpairs, 5), np.inf)

                # Storage for path_info_dict
                Path_Info_Dict_storage = []
                
                # storage for LSP_arrays
                LSP_array_pair_storage = []

                # storage for Year_FP
                Year_FP_pair_storage = []
                
                ##################################
                # Spectrum and fiber assignment  #
                ##################################
                
                for final_K_pair_counter in range(num_kpairs): # Iterate through LAND_pairs
                    
                    # Initialize primary and secondary path info dictionary
                    primary_info_dict = {'destination': -1,
                                         'f_max': [0],
                                         'numHops': 0,
                                         'cost_FP': [0], 
                                         'FP_max': [0], 
                                         'BVT_CBand_count': 0,
                                         'BVT_superCBand_count': 0, 
                                         'BVT_LBand_count': 0, 
                                         'BVT_bitrate_storage' : [], 
                                         'distance': 0}
                    
                    secondary_info_dict = {'destination': -1,
                                         'f_max': [0],
                                         'numHops': 0,
                                         'cost_FP': [0], 
                                         'FP_max': [0], 
                                         'BVT_CBand_count': 0,
                                         'BVT_superCBand_count': 0, 
                                         'BVT_LBand_count': 0, 
                                         'BVT_bitrate_storage' : [], 
                                         'distance': 0}
                    
                    if year == 1:
                        primary_info_dict['links'] = []
                        secondary_info_dict['links'] = []
                    else:
                        primary_info_dict['links'] = self.LAND_Links_Storage[HL_standalone[node_idx]][0] # links from the primary path of previous year
                        secondary_info_dict['links'] = self.LAND_Links_Storage[HL_standalone[node_idx]][1] # links from the secondary path of previous year
                                          
                    self.BVT_type = 1 # start with the BVT with hieghest bitrate value (64-QAM)
                    
                    # make a copy of link-state-profile array for apply changes
                    LSP_array_pair = self.LSP_array.copy()

                    # make a copy of fiber-pair usage array for apply changes
                    Year_FP_pair = self.Year_FP.copy()
                    
                    # Try BVT allocation if the needed traffic is higher than the residual capacity from the last BVT of previous year
                    if HL_pure_throughput_to_assign_primary > 0:
                            
                        # calculate the number of BVTs needed to handle the assigned throughput (Start with 64-QAM BVTs)
                        BVT_number  = int(np.ceil(HL_pure_throughput_to_assign_primary / self.Max_bit_rate_BVT[self.BVT_type - 1]))
                        
                        # Spectrum assignment of primary path
                        primary_path_IDX = int(candidate_path_pair.iloc[final_K_pair_counter]['primary_path_IDx'])
                        primary_info_dict, LSP_array_pair, Year_FP_pair = self._spectrum_assignment(
                                                                                                    path_IDx = primary_path_IDX, 
                                                                                                    path_type = 'primary', 
                                                                                                    year = year, 
                                                                                                    pure_traffic_to_assign = HL_pure_throughput_to_assign_primary,
                                                                                                    BVT_number = BVT_number, 
                                                                                                    K_path_attributes_df = candidate_paths_standalone_df,
                                                                                                    node_IDx = node_idx,
                                                                                                    node_list = HL_standalone,
                                                                                                    GSNR_link = GSNR_opt_link,
                                                                                                    LSP_array_pair = LSP_array_pair, 
                                                                                                    Year_FP_pair = Year_FP_pair, 
                                                                                                    HL_subnet_links = HL_links_indices
                                                                                                    )
                    
                    # Try BVT allocation if the needed traffic is higher than the residual capacity from the last BVT of previous year                        
                    if HL_pure_throughput_to_assign_secondary > 0:
                                            
                        # calculate the number of BVTs needed to handle the assigned throughput (Start with 64-QAM BVTs)
                        BVT_number  = int(np.ceil(HL_pure_throughput_to_assign_secondary / self.Max_bit_rate_BVT[self.BVT_type - 1]))
                                
                        # Spectrum assignment of secondary path
                        secondary_path_IDX = int(candidate_path_pair.iloc[final_K_pair_counter]['secondary_path_IDx'])
                        secondary_info_dict, LSP_array_pair, Year_FP_pair = self._spectrum_assignment(
                                                                                                      path_IDx = secondary_path_IDX, 
                                                                                                      path_type = 'secondary', 
                                                                                                      pure_traffic_to_assign = HL_pure_throughput_to_assign_secondary,
                                                                                                      year = year, 
                                                                                                      BVT_number = BVT_number, 
                                                                                                      K_path_attributes_df = candidate_paths_standalone_df,
                                                                                                      node_IDx = node_idx,
                                                                                                      node_list = HL_standalone,
                                                                                                      GSNR_link = GSNR_opt_link,
                                                                                                      LSP_array_pair = LSP_array_pair, 
                                                                                                      Year_FP_pair = Year_FP_pair, 
                                                                                                      HL_subnet_links = HL_links_indices
                                                                                                      )
                        
                    # Calculate the first cost metric, representing the maximum frequency slot (FS) usage on both primary and secondary paths
                    cost_func[final_K_pair_counter, 0] = max(primary_info_dict['f_max']) + max(secondary_info_dict['f_max'])

                    # Add the number of hops for both primary and secondary paths 
                    cost_func[final_K_pair_counter, 1] = primary_info_dict['numHops'] + secondary_info_dict['numHops']

                    # Reflect the total resource usage considering frequency slots and link lengths
                    cost_func[final_K_pair_counter, 2] = max(primary_info_dict['cost_FP']) + max(secondary_info_dict['cost_FP'])

                    # Placeholder for GSNR cost metric - Initialized with inf 
                    cost_func[final_K_pair_counter, 3] = np.inf

                    # Indicate the maximum frequency path indices used for primary and secondary paths
                    cost_func[final_K_pair_counter, 4] = max(primary_info_dict['FP_max']) + max(secondary_info_dict['FP_max'])

                    # save the link-state-profile (LSP) and fiber-pair (FP) arrays of this LAND pair
                    LSP_array_pair_storage.append(LSP_array_pair.copy())
                    Year_FP_pair_storage.append(Year_FP_pair.copy())
                    
                    # Store the path information dictionary of this LAND pair
                    Path_Info_Dict_storage.append((primary_info_dict, secondary_info_dict))

                # #################### Pair Selection ####################

                # Sort feasible path pairs based on cost function [5 1 2 3 4] in ascending order
                index_feasible_pair = np.lexsort((cost_func[:, 1], cost_func[:, 2], cost_func[:, 0],
                                                cost_func[:, 4], cost_func[:, 3]))  # Sort using lexsort

                # Update the global LSP and Year_FP based on the best LAND pair
                self.LSP_array =  LSP_array_pair_storage[index_feasible_pair[0]]
                self.Year_FP =  Year_FP_pair_storage[index_feasible_pair[0]]
                
                # Select the path information of best LAND pair
                path_pairs_selected = Path_Info_Dict_storage[index_feasible_pair[0]]
                primary_info_dict_selected = path_pairs_selected[0]
                secondary_info_dict_selected = path_pairs_selected[1]
                
                # record the primary and secondary destinations for the selected path
                HL_standalone_dest_profile[year -1, node_idx, 0] = primary_info_dict_selected['destination']
                HL_standalone_dest_profile[year -1, node_idx, 1] = secondary_info_dict_selected['destination']

                # Update the global array of per band BVT count
                self.HL_BVT_number_Cband_annual[year - 1] += primary_info_dict_selected['BVT_CBand_count'] + secondary_info_dict_selected['BVT_CBand_count']
                self.HL_BVT_number_SuperCband_annual[year - 1] += primary_info_dict_selected['BVT_superCBand_count'] + secondary_info_dict_selected['BVT_superCBand_count']
                self.HL_BVT_number_Lband_annual[year - 1] += primary_info_dict_selected['BVT_LBand_count'] + secondary_info_dict_selected['BVT_LBand_count']
                
                
                self.LAND_Links_Storage[HL_standalone[node_idx]] = (
                    primary_info_dict_selected['links'], 
                    secondary_info_dict_selected['links']
                )
                
                # Update the traffic flow on the primary and secondary paths
                self.traffic_flow_links_array[year - 1, primary_info_dict_selected['links']] += 0.5 * HL_needed_traffic
                self.traffic_flow_links_array[year - 1, secondary_info_dict_selected['links']] += 0.5 * HL_needed_traffic
                
                # Initialize the total capacity of activated licenses
                total_cap_LC = 0
                
                # Extract the bitrate storage of established BVT (primary path)
                BVT_bitrate_storage_primary = primary_info_dict_selected['BVT_bitrate_storage']
                
                # If new BVT is establish in this year
                if len(BVT_bitrate_storage_primary) != 0:
                    
                    # Extract the last frequency slots of the established BVTs
                    FS_BVT_storage = primary_info_dict_selected['FS_BVT_storage']
                    
                    # Update per band license trackers based on allocated FS
                    for FS_BVT_counter in range(len(FS_BVT_storage) - 1):
                        total_cap_LC += BVT_bitrate_storage_primary[FS_BVT_counter] # Update total capacity of activated licenses
                        if FS_BVT_storage[FS_BVT_counter] < self.C_band_separation_idx:
                            self.num_100G_licence_CBand_annual[year - 1, HL_standalone[node_idx]] += 4 # Consider 4 licenses for full established BVTs
                        elif self.C_band_separation_idx <= FS_BVT_storage[FS_BVT_counter] < self.supC_band_separation_idx:
                            self.num_100G_licence_superCBand_annual[year - 1, HL_standalone[node_idx]] += 4 # Consider 4 licenses for full established BVTs
                        else:
                            self.num_100G_licence_LBand_annual[year - 1, HL_standalone[node_idx]] += 4 # Consider 4 licenses for full established BVTs
                            
                    # Find type of the last established BVT
                    last_BVT_type_index = np.where(self.Max_bit_rate_BVT == BVT_bitrate_storage_primary[-1])[0] 
                    
                    # Calculate number of licenses in the last BVT
                    num_license_last_BVT = np.ceil((HL_pure_throughput_to_assign_primary - np.sum(BVT_bitrate_storage_primary[:-1])) / self.Ref_license_capacity[last_BVT_type_index])
                    
                    # Update total capacity of activated licenses 
                    total_cap_LC += num_license_last_BVT * self.Ref_license_capacity[last_BVT_type_index] 
                    
                    # Update per band license count based on allocated FS
                    if FS_BVT_storage[-1] < self.C_band_separation_idx:
                        self.num_100G_licence_CBand_annual[year - 1, HL_standalone[node_idx]] += num_license_last_BVT
                    elif self.C_band_separation_idx <= FS_BVT_storage[-1] < self.supC_band_separation_idx:
                        self.num_100G_licence_superCBand_annual[year - 1, HL_standalone[node_idx]] += num_license_last_BVT 
                    else:
                        self.num_100G_licence_LBand_annual[year - 1, HL_standalone[node_idx]] += num_license_last_BVT
                    
                    # Update per band license count for the residual licenses from the last BVT of previous year
                    if year != 1:   
                        if res_license_prev_year_primary_band == 1:
                            self.num_100G_licence_CBand_annual[year - 1, HL_standalone[node_idx]] += num_res_license_prev_year_primary
                        elif res_license_prev_year_primary_band == 2:
                            self.num_100G_licence_superCBand_annual[year - 1, HL_standalone[node_idx]] += num_res_license_prev_year_primary
                        else:
                            self.num_100G_licence_LBand_annual[year - 1, HL_standalone[node_idx]] += num_res_license_prev_year_primary 
                    
                    # Update global number of activated licenses (all bands)
                    self.num_100G_licence_annual[year - 1, HL_standalone[node_idx]] += primary_info_dict_selected['num_added_license'] + num_res_license_prev_year_primary
                    
                    # Update residual capacity of the last activated license in this node 
                    self.Residual_Throughput_LC_standalone_HLs_primary[year - 1, node_idx] = total_cap_LC - HL_pure_throughput_to_assign_primary
                
                # If no new BVT is established in this year 
                else:
                    res_cap_LC_prev_year = self.Residual_Throughput_LC_standalone_HLs_primary[year - 2, node_idx]
                    
                    # If the residual capacity of the last activated licenses from previous year is enough
                    if HL_needed_traffic <= res_cap_LC_prev_year:
                        self.Residual_Throughput_LC_standalone_HLs_primary[year - 1, node_idx] = res_cap_LC_prev_year - HL_needed_traffic
                    else: # If there is a need for more licenses
                        
                        # Find the number of not activated license from the last BVT of this node (previous year)
                        index_primary = np.where(self.Max_bit_rate_BVT == self.last_BVT_type_primary[HL_standalone[node_idx]])[0]
                        num_res_license_prev_year_primary = np.ceil((HL_needed_traffic - res_cap_LC_prev_year) / self.Ref_license_capacity[index_primary])
                        
                        # Update license count based on the allocated band of the last BVT
                        if res_license_prev_year_primary_band == 1:
                            self.num_100G_licence_CBand_annual[year - 1, HL_standalone[node_idx]] += num_res_license_prev_year_primary
                        elif res_license_prev_year_primary_band == 2:
                            self.num_100G_licence_superCBand_annual[year - 1, HL_standalone[node_idx]] += num_res_license_prev_year_primary
                        else:
                            self.num_100G_licence_LBand_annual[year - 1, HL_standalone[node_idx]] += num_res_license_prev_year_primary

                        # Update global number of license tracker (all bands)
                        self.num_100G_licence_annual[year - 1, HL_standalone[node_idx]] += num_res_license_prev_year_primary
                        
                        # Update residual capacity of the last activated licenses in this node 
                        self.Residual_Throughput_LC_standalone_HLs_primary[year - 1, node_idx] =\
                            num_res_license_prev_year_primary * self.Ref_license_capacity[index_primary] + res_cap_LC_prev_year - HL_needed_traffic
                
                # Initialize the total capacity of activated licenses
                total_cap_LC = 0
                
                # Extract the bitrate storage of established BVT (secondary path)
                BVT_bitrate_storage_secondary = secondary_info_dict_selected['BVT_bitrate_storage']
                
                # If new BVT is establish in this year
                if len(BVT_bitrate_storage_secondary) != 0:
                    
                    # Extract the last frequenct slots of the established BVTs
                    FS_BVT_storage = secondary_info_dict_selected['FS_BVT_storage']
                    
                    # Update per band license trackers based on allocated FS
                    for FS_BVT_counter in range(len(FS_BVT_storage) - 1):
                        total_cap_LC += BVT_bitrate_storage_secondary[FS_BVT_counter] # Update total capacity of activated licenses
                        if FS_BVT_storage[FS_BVT_counter] < self.C_band_separation_idx:
                            self.num_100G_licence_CBand_annual[year - 1, HL_standalone[node_idx]] += 4 # Consider 4 licenses for full established BVTs
                        elif self.C_band_separation_idx <= FS_BVT_storage[FS_BVT_counter] < self.supC_band_separation_idx:
                            self.num_100G_licence_superCBand_annual[year - 1, HL_standalone[node_idx]] += 4 # Consider 4 licenses for full established BVTs
                        else: 
                            self.num_100G_licence_LBand_annual[year - 1, HL_standalone[node_idx]] += 4 # Consider 4 licenses for full established BVTs
                            
                    # Find the type of the last established BVT
                    last_BVT_type_index = np.where(self.Max_bit_rate_BVT == BVT_bitrate_storage_secondary[-1])[0] 
                    
                    # Calculate the number of licenses in the last BVT
                    num_license_last_BVT = np.ceil((HL_pure_throughput_to_assign_secondary - np.sum(BVT_bitrate_storage_secondary[:-1])) / self.Ref_license_capacity[last_BVT_type_index])
                    
                    # Update total capacity of activated licenses 
                    total_cap_LC += num_license_last_BVT * self.Ref_license_capacity[last_BVT_type_index]
                    
                    # Update per band license count based on allocated FS
                    if FS_BVT_storage[-1] < self.C_band_separation_idx:
                        self.num_100G_licence_CBand_annual[year - 1, HL_standalone[node_idx]] += num_license_last_BVT
                    elif self.C_band_separation_idx <= FS_BVT_storage[-1] < self.supC_band_separation_idx:
                        self.num_100G_licence_superCBand_annual[year - 1, HL_standalone[node_idx]] += num_license_last_BVT 
                    else:
                        self.num_100G_licence_LBand_annual[year - 1, HL_standalone[node_idx]] += num_license_last_BVT
                    
                    # Update per band license count for the residual licenses from the last BVT of previous year
                    if year != 1:   
                        if res_license_prev_year_secondary_band == 1:
                            self.num_100G_licence_CBand_annual[year - 1, HL_standalone[node_idx]] += num_res_license_prev_year_secondary
                        elif res_license_prev_year_secondary_band == 2:
                            self.num_100G_licence_superCBand_annual[year - 1, HL_standalone[node_idx]] += num_res_license_prev_year_secondary
                        else:
                            self.num_100G_licence_LBand_annual[year - 1, HL_standalone[node_idx]] += num_res_license_prev_year_secondary 
                            
                    # Update global number of activated licenses (all bands)  
                    self.num_100G_licence_annual[year - 1, HL_standalone[node_idx]] += secondary_info_dict_selected['num_added_license'] + num_res_license_prev_year_secondary
                    
                    # Update residual capacity of the last activated license in this node 
                    self.Residual_Throughput_LC_standalone_HLs_secondary[year - 1, node_idx] = total_cap_LC - HL_pure_throughput_to_assign_secondary

                # If no new BVT is established in this year
                else:
                    res_cap_LC_prev_year = self.Residual_Throughput_LC_standalone_HLs_secondary[year - 2, node_idx]
                    
                    # If the residual capacity of the last activated license from previous year is enough
                    if HL_needed_traffic <= res_cap_LC_prev_year:
                        self.Residual_Throughput_LC_standalone_HLs_secondary[year - 1, node_idx] = res_cap_LC_prev_year - HL_needed_traffic
                    else:
                        
                        # Find the number of not activated license from the last BVT of this node (previous year)
                        index_secondary = np.where(self.Max_bit_rate_BVT == self.last_BVT_type_primary[HL_standalone[node_idx]])[0]
                        num_res_license_prev_year_secondary = np.ceil((HL_needed_traffic - res_cap_LC_prev_year) / self.Ref_license_capacity[index_secondary])
                        
                        # Update license count based on the allocated band of the last BVT
                        if res_license_prev_year_secondary_band == 1:
                            self.num_100G_licence_CBand_annual[year - 1, HL_standalone[node_idx]] += num_res_license_prev_year_secondary
                        elif res_license_prev_year_secondary_band == 2:
                            self.num_100G_licence_superCBand_annual[year - 1, HL_standalone[node_idx]] += num_res_license_prev_year_secondary
                        else:
                            self.num_100G_licence_LBand_annual[year - 1, HL_standalone[node_idx]] += num_res_license_prev_year_secondary

                        # Update global number of license tracker (all bands)
                        self.num_100G_licence_annual[year - 1, HL_standalone[node_idx]] += num_res_license_prev_year_secondary
                        
                        # Update residual capacity of the last activated licenses in this node 
                        self.Residual_Throughput_LC_standalone_HLs_secondary[year - 1, node_idx] =\
                            num_res_license_prev_year_secondary * self.Ref_license_capacity[index_secondary] + res_cap_LC_prev_year - HL_needed_traffic

                # Due to the same paths in all years, so the latecny stores just in the first year with all node need new BVT establishment
                if year == 1:
                    
                    # Store latency and destinations of this node (primary_path, secondary_path)
                    self.destinations_storage[HL_standalone[node_idx]] = [primary_info_dict_selected['destination'], 
                                                                          secondary_info_dict_selected['destination']]
                    
                    self.path_latency_storage[HL_standalone[node_idx]] = (primary_info_dict_selected['distance'] * 5, 
                                                                          secondary_info_dict_selected['distance'] * 5)

                if year > 1 and (hierarchy_level == minimum_level or HL_needed_traffic != 0):

                    # check if the required HL traffic exceeds the residual BVT throughput from the previous year (primary path)
                    if HL_needed_traffic > self.Residual_Throughput_BVT_standalone_HLs_primary[year - 2, node_idx]:

                        # Update the residual capacity of the last established BVT in this year
                        self.Residual_Throughput_BVT_standalone_HLs_primary[year - 1, node_idx] =\
                            self.Residual_Throughput_BVT_standalone_HLs_primary[year - 2, node_idx] + \
                                np.sum(BVT_bitrate_storage_primary) - HL_needed_traffic
                    
                        # Update node_capacity_profile for destination of primary path (previous year)
                        node_capacity_profile[HL_standalone_dest_profile[year - 2, node_idx, 0], node_cap_update_idx] += 0.5 * self.Residual_Throughput_BVT_standalone_HLs_primary[year - 2, node_idx]

                        # Update node_capacity_profile for destination of primary path (this year)
                        node_capacity_profile[HL_standalone_dest_profile[year - 1, node_idx, 0], node_cap_update_idx] += 0.5 * (HL_needed_traffic - self.Residual_Throughput_BVT_standalone_HLs_primary[year - 2, node_idx])
                        

                    # if residual capacity is enough, just subtracts the required traffic from the existing capacity
                    else:
                        
                        # deduct the required HL traffic from the previous year's residual throughput.
                        self.Residual_Throughput_BVT_standalone_HLs_primary[year - 1, node_idx] = self.Residual_Throughput_BVT_standalone_HLs_primary[year - 2, node_idx] - HL_needed_traffic
                
                        # maintain the same destination profile as the previous year (no change in destination node)
                        HL_standalone_dest_profile[year - 1, node_idx, 0] = HL_standalone_dest_profile[year - 2, node_idx, 0]
                
                        # add half of the needed traffic to the source node's allocated capacity.
                        node_capacity_profile[HL_standalone_dest_profile[year - 1, node_idx, 0], node_cap_update_idx] += 0.5 * HL_needed_traffic
                        
                    
                    # check if the required HL traffic exceeds the residual BVT throughput from the previous year (secondary path)
                    if HL_needed_traffic > self.Residual_Throughput_BVT_standalone_HLs_secondary[year - 2, node_idx]:
                                                
                        # Update the residual capacity of the last established BVT in this year
                        self.Residual_Throughput_BVT_standalone_HLs_secondary[year - 1, node_idx] =\
                            self.Residual_Throughput_BVT_standalone_HLs_secondary[year - 2, node_idx] + \
                                np.sum(BVT_bitrate_storage_secondary) - HL_needed_traffic
                        
                        # Update node_capacity_profile for destination of secondary path (previous year)
                        node_capacity_profile[HL_standalone_dest_profile[year - 2, node_idx, 1], node_cap_update_idx] += 0.5 * self.Residual_Throughput_BVT_standalone_HLs_secondary[year - 2, node_idx]

                        # Update node_capacity_profile for destination of secondary path (this year)
                        node_capacity_profile[HL_standalone_dest_profile[year - 1, node_idx, 1], node_cap_update_idx] += 0.5 * (HL_needed_traffic - self.Residual_Throughput_BVT_standalone_HLs_secondary[year - 2, node_idx])


                    # if residual capacity is enough, just subtracts the required traffic from the existing capacity
                    else:
                        
                        # deduct the required HL traffic from the previous year's residual throughput.
                        self.Residual_Throughput_BVT_standalone_HLs_secondary[year - 1, node_idx] = self.Residual_Throughput_BVT_standalone_HLs_secondary[year - 2, node_idx] -\
                            HL_needed_traffic
                
                        # maintain the same destination profile as the previous year (no change in destination node)
                        HL_standalone_dest_profile[year - 1, node_idx, 1] = HL_standalone_dest_profile[year - 2, node_idx, 1]
                                        
                        # add half of the needed traffic to the source node's allocated capacity.
                        node_capacity_profile[HL_standalone_dest_profile[year - 1, node_idx, 1], node_cap_update_idx] += 0.5 * HL_needed_traffic
                                
                # if this is the first year
                elif hierarchy_level == minimum_level or HL_needed_traffic != 0:
                    
                    # Update the residual capacity of the last established BVT in this year (primary and secondary path)
                    self.Residual_Throughput_BVT_standalone_HLs_primary[0, node_idx] = np.sum(BVT_bitrate_storage_primary) - HL_needed_traffic
                    self.Residual_Throughput_BVT_standalone_HLs_secondary[0, node_idx] = np.sum(BVT_bitrate_storage_secondary) - HL_needed_traffic
                
                    # Update node_capacity_profile for destination of primary path (this year)
                    node_capacity_profile[HL_standalone_dest_profile[year - 1, node_idx, 0], node_cap_update_idx] += 0.5 * node_capacity_profile[HL_standalone[node_idx], node_cap_update_idx + 1]
                
                    # Update node_capacity_profile for destination of secondary path (this year)
                    node_capacity_profile[HL_standalone_dest_profile[year - 1, node_idx, 1], node_cap_update_idx] += 0.5 * node_capacity_profile[HL_standalone[node_idx], node_cap_update_idx + 1]

            #######################################################
            # Part 2: Spectrum assignment for colocated HL nodes
            #######################################################

            # Initialize the cost function matrix with infinity values for each metric (f_max, N_hop, cost, GSNR, FP_max)
            cost_func = np.inf * np.ones(shape = (1, 5))

            # number of precalculated k-shortest path of colocated nodes
            max_path_secondary = candidate_paths_colocated_df.groupby('src_node').size().to_numpy()

            for node_idx in range(len(HL_colocated)): # Iterate through colocated nodes
                
                # Initialize residual and added licenses (primary and secondary paths) of this node
                num_res_license_prev_year_primary = 0
                self.num_added_license_this_year_primary = 0
                num_res_license_prev_year_secondary = 0
                self.num_added_license_this_year_secondary = 0
                
                print(f"Processing colocated node {HL_colocated[node_idx]}")
                
                # get traffic demand for this node in this year
                if hierarchy_level == minimum_level:
                    HL_needed_traffic = self.lowest_HL_added_traffic_annual_colocated[year - 1, node_idx]
                else:
                    HL_needed_traffic = node_capacity_profile[HL_colocated[node_idx], node_cap_update_idx + 1]
                
                if year != 1: # if it isnt the first year
                    
                    # subtract residual throughput (unallocated traffic from previous years) 
                    HL_pure_throughput_to_assign_primary = HL_needed_traffic - self.Residual_Throughput_BVT_colocated_HLs_primary[year - 2, node_idx] # primary path
                    HL_pure_throughput_to_assign_secondary = HL_needed_traffic - self.Residual_Throughput_BVT_colocated_HLs_secondary[year - 2, node_idx] # secondary path
                
                else: # if it is the first year    
                    HL_pure_throughput_to_assign_primary = HL_needed_traffic
                    HL_pure_throughput_to_assign_secondary = HL_needed_traffic
                
                if hierarchy_level == minimum_level:
                    
                    # store traffic capacity assigned to current node
                    node_capacity_profile[HL_colocated[node_idx], node_cap_update_idx + 1] = HL_needed_traffic
                    
                
                if year != 1:
                    
                    # calculate the number of not activated licenses of the last established BVT from prevoous year
                    Residual_cap_primary_prev_year = np.round(self.Residual_Throughput_BVT_colocated_HLs_primary[year - 2, node_idx] - self.Residual_Throughput_LC_colocated_HLs_primary[year - 2, node_idx])
                    Residual_cap_secondary_prev_year = np.round(self.Residual_Throughput_BVT_colocated_HLs_secondary[year - 2, node_idx] - self.Residual_Throughput_LC_colocated_HLs_secondary[year - 2, node_idx])                    
                    index_primary = np.where(self.Max_bit_rate_BVT == self.last_BVT_type_primary[HL_colocated[node_idx]])[0] # type of the last BVT of primary path
                    index_secondary = np.where(self.Max_bit_rate_BVT == self.last_BVT_type_secondary[HL_colocated[node_idx]])[0] # type of the last BVT of secondary path
                    num_res_license_prev_year_primary = int(Residual_cap_primary_prev_year / self.Ref_license_capacity[index_primary])
                    res_license_prev_year_primary_band = self.last_BVT_Band_primary[HL_colocated[node_idx]] # Specify the band in which the last established BVT is assigned (primary)
                    num_res_license_prev_year_secondary = int(Residual_cap_secondary_prev_year / self.Ref_license_capacity[index_secondary])
                    res_license_prev_year_secondary_band = self.last_BVT_Band_secondary[HL_colocated[node_idx]] # Specify the band in which the last established BVT is assigned (secondary)
             
                
                if HL_pure_throughput_to_assign_primary > 0 and HL_pure_throughput_to_assign_secondary <= 0:
                    print('one path need to be assigned')
                elif HL_pure_throughput_to_assign_primary <= 0 and HL_pure_throughput_to_assign_secondary > 0:
                    print('one path need to be assigned')
                    
                                    
                ##################################
                # Spectrum and fiber assignment  #
                ##################################
                
                self.BVT_type = 1 # Always use the BVT with hieghest bitrate value (64-QAM) for primary colocated path
                
                if HL_pure_throughput_to_assign_primary > 0:
                
                    # calculate the number of BVTs needed to handle the assigned throughput (Start with 64-QAM BVTs)
                    BVT_number  = int(np.ceil(HL_pure_throughput_to_assign_primary / self.Max_bit_rate_BVT[self.BVT_type - 1]))

                    # Spectrum assignment for primary path
                    Year_FP_HL_colocated, BVT_bitrate_storage_primary, FS_BVT_storage = self._spectrum_assignment(
                                                                                                                  path_IDx = None,
                                                                                                                  path_type = None,
                                                                                                                  year = year, 
                                                                                                                  K_path_attributes_df = candidate_paths_colocated_df,
                                                                                                                  BVT_number = BVT_number,
                                                                                                                  node_IDx = node_idx,
                                                                                                                  pure_traffic_to_assign = HL_pure_throughput_to_assign_primary,
                                                                                                                  node_list = HL_colocated,
                                                                                                                  GSNR_link = GSNR_opt_link,
                                                                                                                  LSP_array_pair = None, 
                                                                                                                  Year_FP_pair = None, 
                                                                                                                  HL_subnet_links = None
                                                                                                                  )
                    # Update global array of Year_FP_HL_colocated
                    self.Year_FP_HL_colocated = Year_FP_HL_colocated
                    
                    # Initialize the total capacity of activated licenses
                    total_cap_LC = 0
                    
                    # Update per band license trackers based on allocated FS
                    for FS_BVT_counter in range(len(FS_BVT_storage) - 1):
                        total_cap_LC += BVT_bitrate_storage_primary[FS_BVT_counter] # Update total capacity of activated licenses
                        if FS_BVT_storage[FS_BVT_counter] < self.C_band_separation_idx:
                            self.num_100G_licence_CBand_annual[year - 1, HL_colocated[node_idx]] += 4 # Consider 4 licenses for full established BVTs
                        elif self.C_band_separation_idx <= FS_BVT_storage[FS_BVT_counter] < self.supC_band_separation_idx:
                            self.num_100G_licence_superCBand_annual[year - 1, HL_colocated[node_idx]] += 4 # Consider 4 licenses for full established BVTs
                        else:
                            self.num_100G_licence_LBand_annual[year - 1, HL_colocated[node_idx]] += 4 # Consider 4 licenses for full established BVTs
                    
                    # Find type of the last established BVT
                    last_BVT_type_index = np.where(self.Max_bit_rate_BVT == BVT_bitrate_storage_primary[-1])[0] 
                    
                    # Calculate number of licenses in the last BVT
                    num_license_last_BVT = np.ceil((HL_pure_throughput_to_assign_primary - np.sum(BVT_bitrate_storage_primary[:-1])) / self.Ref_license_capacity[last_BVT_type_index])
                    
                    # Update total capacity of activated licenses 
                    total_cap_LC += num_license_last_BVT * self.Ref_license_capacity[last_BVT_type_index]
                    
                    # Update per band license count based on allocated FS
                    if FS_BVT_storage[-1] < self.C_band_separation_idx:
                        self.num_100G_licence_CBand_annual[year - 1, HL_colocated[node_idx]] += num_license_last_BVT
                    elif self.C_band_separation_idx <= FS_BVT_storage[-1] < self.supC_band_separation_idx:
                        self.num_100G_licence_superCBand_annual[year - 1, HL_colocated[node_idx]] += num_license_last_BVT 
                    else:
                        self.num_100G_licence_LBand_annual[year - 1, HL_colocated[node_idx]] += num_license_last_BVT
                    
                    # Update per band license count for the residual licenses from the last BVT of previous year
                    if year != 1:   
                        if res_license_prev_year_primary_band == 1:
                            self.num_100G_licence_CBand_annual[year - 1, HL_colocated[node_idx]] += num_res_license_prev_year_primary
                        elif res_license_prev_year_primary_band == 2:
                            self.num_100G_licence_superCBand_annual[year - 1, HL_colocated[node_idx]] += num_res_license_prev_year_primary
                        else:
                            self.num_100G_licence_LBand_annual[year - 1, HL_colocated[node_idx]] += num_res_license_prev_year_primary 
                    
                    # Update global number of activated licenses (all bands)
                    self.num_100G_licence_annual[year - 1, HL_colocated[node_idx]] += self.num_added_license_this_year_primary + num_res_license_prev_year_primary
                    
                    # Update residual capacity of the last activated license in this node 
                    self.Residual_Throughput_LC_colocated_HLs_primary[year - 1, node_idx] = total_cap_LC - HL_pure_throughput_to_assign_primary
                
                # If no new BVT is established in this year  
                else:
                    
                    res_cap_LC_prev_year = self.Residual_Throughput_LC_colocated_HLs_primary[year - 2, node_idx]
                    
                    # If the residual capacity of the last activated licenses from previous year is enough
                    if HL_needed_traffic <= res_cap_LC_prev_year:
                        self.Residual_Throughput_LC_colocated_HLs_primary[year - 1, node_idx] = res_cap_LC_prev_year - HL_needed_traffic
                    else: # If there is a need for more licenses
                        
                        # Find the number of not activated license from the last BVT of this node (previous year)
                        index_primary = 0 # Always use BVT with highest bitrate in primary colocated path
                        num_res_license_prev_year_primary = np.ceil((HL_needed_traffic - res_cap_LC_prev_year) / self.Ref_license_capacity[index_primary])
                        
                        # Update license count based on the allocated band of the last BVT
                        if res_license_prev_year_primary_band == 1:
                            self.num_100G_licence_CBand_annual[year - 1, HL_colocated[node_idx]] += num_res_license_prev_year_primary
                        elif res_license_prev_year_primary_band == 2:
                            self.num_100G_licence_superCBand_annual[year - 1, HL_colocated[node_idx]] += num_res_license_prev_year_primary
                        else:
                            self.num_100G_licence_LBand_annual[year - 1, HL_colocated[node_idx]] += num_res_license_prev_year_primary
                     
                        # Update global number of license tracker (all bands)
                        self.num_100G_licence_annual[year - 1, HL_colocated[node_idx]] += num_res_license_prev_year_primary
                        
                        # Update residual capacity of the last activated licenses in this node 
                        self.Residual_Throughput_LC_colocated_HLs_primary[year - 1, node_idx] =\
                            num_res_license_prev_year_primary * self.Ref_license_capacity[index_primary] + res_cap_LC_prev_year - HL_needed_traffic
                               
                # Calculate the number of k-shortest path for this colocated node (secondary path)
                num_kpairs = int(min(max_path_secondary[node_idx], kpair_colocated))
                cost_func = np.full((num_kpairs, 5), np.inf)  # Initialize cost function with infinity

                # storage for LSP_arrays
                LSP_array_pair_storage = []

                # storage for Year_FP
                Year_FP_pair_storage = []
                                
                # Storage for path_info_dict
                Path_Info_Dict_storage = []
                
                for final_K_pair_counter in range(num_kpairs): # Iterate through all candidate LAND_pairs
                    
                     # Initialize secondary path info dictionary
                    secondary_info_dict = {'destination': -1,
                                         'links': [], 
                                         'f_max': [0],
                                         'numHops': 0,
                                         'cost_FP': [0], 
                                         'FP_max': [0], 
                                         'BVT_CBand_count': 0,
                                         'BVT_superCBand_count': 0, 
                                         'BVT_LBand_count': 0, 
                                         'BVT_bitrate_storage' : []}

                    self.BVT_type = 1 # start with the BVT with hieghest bitrate value (64-QAM)
                    
                    # Try BVT allocation if the needed traffic is higher than the residual capacity from the last BVT of previous year
                    if HL_pure_throughput_to_assign_secondary > 0:
                    
                        # calculate the number of BVTs needed to handle the assigned throughput
                        BVT_number  = int(np.ceil(HL_pure_throughput_to_assign_secondary / self.Max_bit_rate_BVT[self.BVT_type - 1]))
                        
                        # make a copy of link-state-profile array for apply changes
                        LSP_array_pair = self.LSP_array.copy()
                        
                        # make a copy of fiber-pair usage array for apply changes
                        Year_FP_pair = self.Year_FP.copy()

                        # Spectrum assignment for secondary path
                        secondary_path_IDX = candidate_paths_colocated_df[candidate_paths_colocated_df['src_node'] == HL_colocated[node_idx]].head(1).index[0]
                        secondary_info_dict, LSP_array_pair, Year_FP_pair = self._spectrum_assignment(
                                                                                                      path_IDx = secondary_path_IDX, 
                                                                                                      path_type = 'secondary', 
                                                                                                      year = year, 
                                                                                                      pure_traffic_to_assign = HL_pure_throughput_to_assign_secondary,
                                                                                                      BVT_number = BVT_number, 
                                                                                                      K_path_attributes_df = candidate_paths_colocated_df,
                                                                                                      node_IDx = node_idx,
                                                                                                      node_list = HL_colocated,
                                                                                                      GSNR_link = GSNR_opt_link,
                                                                                                      LSP_array_pair = LSP_array_pair, 
                                                                                                      Year_FP_pair = Year_FP_pair, 
                                                                                                      HL_subnet_links = HL_links_indices
                                                                                                      )
                                                
                    # Calculate the first cost metric, representing the maximum frequency slot (FS) usage on both primary and secondary paths
                    cost_func[final_K_pair_counter, 0] = max(secondary_info_dict['f_max'])

                    # Add the number of hops for both primary and secondary paths 
                    cost_func[final_K_pair_counter, 1] = secondary_info_dict['numHops']

                    # Reflect the total resource usage considering frequency slots and link lengths
                    cost_func[final_K_pair_counter, 2] = max(secondary_info_dict['cost_FP'])

                    # Placeholder for GSNR cost metric - Initialized with inf 
                    cost_func[0, 3] = np.inf

                    # Indicate the maximum frequency path indices used for primary and secondary paths
                    cost_func[final_K_pair_counter, 4] = max(secondary_info_dict['FP_max'])

                    # save the link-state-profile (LSP) and fiber-pair (FP) arrays for further evaluation
                    LSP_array_pair_storage.append(LSP_array_pair.copy())
                    Year_FP_pair_storage.append(Year_FP_pair.copy())

                    # Store the path information dictionary of this secondary path
                    Path_Info_Dict_storage.append(secondary_info_dict)

                # #################### Pair Selection ####################

                # Sort feasible path pairs based on cost function [5 1 2 3 4] in ascending order
                index_feasible_pair = np.lexsort((cost_func[:, 1], cost_func[:, 2], cost_func[:, 0],
                                                cost_func[:, 4], cost_func[:, 3]))  # Sort using lexsort

                # Update the global LSP and Year_FP based on the best LAND pair
                self.LSP_array =  LSP_array_pair_storage[index_feasible_pair[0]]
                self.Year_FP =  Year_FP_pair_storage[index_feasible_pair[0]]
                
                # Select the path information of best secondary path
                secondary_info_dict_selected = Path_Info_Dict_storage[index_feasible_pair[0]]

                # record the secondary destinations for the selected path
                HL_colocated_dest_profile[year -1, node_idx] = secondary_info_dict_selected['destination']

                # Update the global array of per band BVT count
                self.HL_BVT_number_Cband_annual[year - 1] += secondary_info_dict_selected['BVT_CBand_count']
                self.HL_BVT_number_SuperCband_annual[year - 1] += secondary_info_dict_selected['BVT_superCBand_count']
                self.HL_BVT_number_Lband_annual[year - 1] += secondary_info_dict_selected['BVT_LBand_count']
                
                # Update the traffic flow on the secondary path
                self.traffic_flow_links_array[year - 1, secondary_info_dict_selected['links']] += 0.5 * HL_needed_traffic
                
                # Initialize the total capacity of activated licenses
                total_cap_LC = 0
                
                # Extract the bitrate storage of established BVT (secondary path)
                BVT_bitrate_storage_secondary = secondary_info_dict_selected['BVT_bitrate_storage']
                
                # If new BVT is establish in this year
                if len(BVT_bitrate_storage_secondary) != 0:
                    
                    # Extract the last frequency slots of the established BVTs
                    FS_BVT_storage = secondary_info_dict_selected['FS_BVT_storage']
                    
                    # Update per band license trackers based on allocated FS
                    for FS_BVT_counter in range(len(FS_BVT_storage) - 1):
                        total_cap_LC += BVT_bitrate_storage_secondary[FS_BVT_counter] # Update total capacity of activated licenses
                        if FS_BVT_storage[FS_BVT_counter] < self.C_band_separation_idx:
                            self.num_100G_licence_CBand_annual[year - 1, HL_colocated[node_idx]] += 4 # Consider 4 licenses for full established BVTs
                        elif self.C_band_separation_idx <= FS_BVT_storage[FS_BVT_counter] < self.supC_band_separation_idx:
                            self.num_100G_licence_superCBand_annual[year - 1, HL_colocated[node_idx]] += 4 # Consider 4 licenses for full established BVTs
                        else:
                            self.num_100G_licence_LBand_annual[year - 1, HL_colocated[node_idx]] += 4 # Consider 4 licenses for full established BVTs
                    
                    # Find type of the last established BVT
                    last_BVT_type_index = np.where(self.Max_bit_rate_BVT == BVT_bitrate_storage_secondary[-1])[0] 
                    
                    # Calculate number of licenses in the last BVT
                    num_license_last_BVT = np.ceil((HL_pure_throughput_to_assign_secondary - np.sum(BVT_bitrate_storage_secondary[:-1])) / self.Ref_license_capacity[last_BVT_type_index])
                    
                    # Update total capacity of activated licenses 
                    total_cap_LC += num_license_last_BVT * self.Ref_license_capacity[last_BVT_type_index]
                    
                    # Update per band license count based on allocated FS
                    if FS_BVT_storage[-1] < self.C_band_separation_idx:
                        self.num_100G_licence_CBand_annual[year - 1, HL_colocated[node_idx]] += num_license_last_BVT
                    elif self.C_band_separation_idx <= FS_BVT_storage[-1] < self.supC_band_separation_idx:
                        self.num_100G_licence_superCBand_annual[year - 1, HL_colocated[node_idx]] += num_license_last_BVT 
                    else:
                        self.num_100G_licence_LBand_annual[year - 1, HL_colocated[node_idx]] += num_license_last_BVT
                    
                    # Update per band license count for the residual licenses from the last BVT of previous year
                    if year != 1:   
                        if res_license_prev_year_secondary_band == 1:
                            self.num_100G_licence_CBand_annual[year - 1, HL_colocated[node_idx]] += num_res_license_prev_year_secondary
                        elif res_license_prev_year_secondary_band == 2:
                            self.num_100G_licence_superCBand_annual[year - 1, HL_colocated[node_idx]] += num_res_license_prev_year_secondary
                        else:
                            self.num_100G_licence_LBand_annual[year - 1, HL_colocated[node_idx]] += num_res_license_prev_year_secondary 
                    
                    # Update global number of activated licenses (all bands)
                    self.num_100G_licence_annual[year - 1, HL_colocated[node_idx]] += secondary_info_dict_selected['num_added_license'] + num_res_license_prev_year_secondary
                    
                    # Update residual capacity of the last activated license in this node 
                    self.Residual_Throughput_LC_colocated_HLs_secondary[year - 1, node_idx] = total_cap_LC - HL_pure_throughput_to_assign_secondary
                
                # If no new BVT is established in this year     
                else:
                    res_cap_LC_prev_year = self.Residual_Throughput_LC_colocated_HLs_secondary[year - 2, node_idx]
                    
                     # If the residual capacity of the last activated licenses from previous year is enough
                    if HL_needed_traffic <= res_cap_LC_prev_year:
                        self.Residual_Throughput_LC_colocated_HLs_secondary[year - 1, node_idx] = res_cap_LC_prev_year - HL_needed_traffic
                    else: # If there is a need for more licenses
                        
                        # Find the number of not activated license from the last BVT of this node (previous year)
                        index_secondary = np.where(self.Max_bit_rate_BVT == self.last_BVT_type_secondary[HL_colocated[node_idx]])[0]
                        num_res_license_prev_year_secondary = np.ceil((HL_needed_traffic - res_cap_LC_prev_year) / self.Ref_license_capacity[index_secondary])
                        
                        # Update license count based on the allocated band of the last BVT
                        if res_license_prev_year_secondary_band == 1:
                            self.num_100G_licence_CBand_annual[year - 1, HL_colocated[node_idx]] += num_res_license_prev_year_secondary
                        elif res_license_prev_year_secondary_band == 2:
                            self.num_100G_licence_superCBand_annual[year - 1, HL_colocated[node_idx]] += num_res_license_prev_year_secondary
                        else:
                            self.num_100G_licence_LBand_annual[year - 1, HL_colocated[node_idx]] += num_res_license_prev_year_secondary

                        # Update global number of license tracker (all bands)
                        self.num_100G_licence_annual[year - 1, HL_colocated[node_idx]] += num_res_license_prev_year_secondary
                        
                        # Update residual capacity of the last activated licenses in this node 
                        self.Residual_Throughput_LC_colocated_HLs_secondary[year - 1, node_idx] =\
                            num_res_license_prev_year_secondary * self.Ref_license_capacity[index_secondary] + res_cap_LC_prev_year - HL_needed_traffic
                
    
                if year > 1 and (hierarchy_level == minimum_level or HL_needed_traffic != 0):

                    # check if the required HL traffic exceeds the residual BVT throughput from the previous year (primary path)
                    if HL_needed_traffic > self.Residual_Throughput_BVT_colocated_HLs_primary[year - 2, node_idx]:
                        
                        # Update the residual capacity of the last established BVT in this year
                        self.Residual_Throughput_BVT_colocated_HLs_primary[year - 1, node_idx] =\
                            self.Residual_Throughput_BVT_colocated_HLs_primary[year - 2, node_idx] +\
                                np.sum(BVT_bitrate_storage_primary) - HL_needed_traffic

                        # Update node_capacity_profile for destination of primary path (previous year)
                        # Note: the destination of colocated nodes (primary path) are their site
                        node_capacity_profile[HL_colocated[node_idx], node_cap_update_idx] += 0.5 * self.Residual_Throughput_BVT_colocated_HLs_primary[year - 2, node_idx]

                        # Update node_capacity_profile for destination if primary path (this year)
                        # Note: the destination of colocated nodes (primary path) are their site
                        node_capacity_profile[HL_colocated[node_idx], node_cap_update_idx] += 0.5 * (HL_needed_traffic - self.Residual_Throughput_BVT_colocated_HLs_primary[year - 2, node_idx])

                    # if residual capacity is enough, just subtracts the required traffic from the existing capacity
                    else:
                        
                        # deduct the required HL traffic from the previous year's residual throughput.
                        self.Residual_Throughput_BVT_colocated_HLs_primary[year - 1, node_idx] = self.Residual_Throughput_BVT_colocated_HLs_primary[year - 2, node_idx] - HL_needed_traffic
                
                        # add half of the needed traffic to the source node's allocated capacity.
                        node_capacity_profile[HL_colocated[node_idx], node_cap_update_idx] += 0.5 * HL_needed_traffic
                        
                    # check if the required HL traffic exceeds the residual BVT throughput from the previous year (secondary path)
                    if HL_needed_traffic > self.Residual_Throughput_BVT_colocated_HLs_secondary[year - 2, node_idx]:

                        # Update the residual capacity of the last established BVT in this year                        
                        self.Residual_Throughput_BVT_colocated_HLs_secondary[year - 1, node_idx] =\
                            self.Residual_Throughput_BVT_colocated_HLs_secondary[year - 2, node_idx] +\
                                np.sum(BVT_bitrate_storage_secondary) - HL_needed_traffic
                        
                                    
                        # Update node_capacity_profile for destination if secondary path (previous year)
                        node_capacity_profile[HL_colocated_dest_profile[year - 1, node_idx], node_cap_update_idx] += 0.5 * (self.Residual_Throughput_BVT_colocated_HLs_secondary[year - 2, node_idx])
                                            
                        # Update node_capacity_profile for destination if secondary path (this year)
                        node_capacity_profile[HL_colocated_dest_profile[year - 1, node_idx], node_cap_update_idx] += 0.5 * (HL_needed_traffic - self.Residual_Throughput_BVT_colocated_HLs_secondary[year - 2, node_idx])


                    # if residual capacity is enough, just subtracts the required traffic from the existing capacity
                    else:
                        
                        # deduct the required HL traffic from the previous year's residual throughput.
                        self.Residual_Throughput_BVT_colocated_HLs_secondary[year - 1, node_idx] = self.Residual_Throughput_BVT_colocated_HLs_secondary[year - 2, node_idx] -\
                            HL_needed_traffic
                
                        # maintain the same destination profile as the previous year (no change in destination node).
                        HL_colocated_dest_profile[year - 1, node_idx] = HL_colocated_dest_profile[year - 2, node_idx]
                                
                        # add the other half of the needed traffic to the destination node's allocated capacity.
                        node_capacity_profile[HL_colocated_dest_profile[year - 1, node_idx], node_cap_update_idx] += 0.5 * HL_needed_traffic
                        

                # if this is the first year
                elif hierarchy_level == minimum_level or HL_needed_traffic != 0:

                    # Update the residual capacity of the last established BVT in this year (primary and secondary path)
                    self.Residual_Throughput_BVT_colocated_HLs_primary[0, node_idx] = np.sum(BVT_bitrate_storage_primary) - HL_needed_traffic
                    self.Residual_Throughput_BVT_colocated_HLs_secondary[0, node_idx] = np.sum(BVT_bitrate_storage_secondary) - HL_needed_traffic
                
                    # update source node capacity: add half of the node's original capacity (from the capacity profile) to the allocated capacity.
                    node_capacity_profile[HL_colocated[node_idx], node_cap_update_idx] += 0.5 * node_capacity_profile[HL_colocated[node_idx], node_cap_update_idx + 1]
                
                    # update destination node capacity: add the remaining half of the node's original capacity to the destination node's allocated capacity.
                    node_capacity_profile[HL_colocated_dest_profile[year - 1, node_idx], node_cap_update_idx] += 0.5 * node_capacity_profile[HL_colocated[node_idx], node_cap_update_idx + 1]



            ######################################################################
            # Update Fiber-Pairs (FP) and Degree Counters for Each Year
            ######################################################################
            
            if year > 1:

                #  update Frequency Plan (FP) for HL4 SubNetwork Links
                for link_idx in range(len(HL_subnet_links)):

                        # If the FP for the current year and link is not established (i.e., equals zero) inherit the FP from the previous year for continuity.
                        if hierarchy_level == minimum_level and self.Year_FP[year - 2, HL_links_indices[link_idx]] == 0 and self.Year_FP[year - 1, HL_links_indices[link_idx]] == 0:
                            self.Year_FP[year - 1, HL_links_indices[link_idx]] = self.Year_FP[year - 2, HL_links_indices[link_idx]]
                        elif self.Year_FP[year - 1, HL_links_indices[link_idx]] == 0:
                            self.Year_FP[year - 1, HL_links_indices[link_idx]] = self.Year_FP[year - 2, HL_links_indices[link_idx]]


                # update Frequency Plan (FP) for HL4 Co-located Links
                for node_idx in range(len(HL_colocated)):

                    # If the FP for the current year and link is not established (i.e., equals zero) inherit the FP from the previous year for continuity.
                    if hierarchy_level == minimum_level and self.Year_FP_HL_colocated[year - 1, node_idx] == 0 and self.Year_FP_HL_colocated[year - 2, node_idx] == 0:
                        self.Year_FP_HL_colocated[year - 1, node_idx] = self.Year_FP_HL_colocated[year - 2, node_idx]
                    elif self.Year_FP_HL_colocated[year - 1, node_idx] == 0:
                        self.Year_FP_HL_colocated[year - 1, node_idx] = self.Year_FP_HL_colocated[year - 2, node_idx]

            ##################################################
            # Calculate Total Effective Fiber-Pairs (FP)
            ##################################################

            # Compute the weighted total FP for the current year: 
            # - First term: Weighted sum of FP across all links using provided link weights.
            # - Second term: Contribution from colocated HL links is multiplied by zeros, effectively ignoring them.
            self.Total_effective_FP[year - 1] = 2 * np.dot(self.Year_FP[year - 1, :], self.network.weights_array.T) + \
                0 * 2 * np.dot(self.Year_FP_HL_colocated[year - 1, :], 0.5 * np.ones(len(HL_colocated)))

            # Save Node Capacity Profile for Current Year
            self.node_capacity_profile_array[year - 1] = node_capacity_profile

            #############################################################
            # Fiber-Pairs (FP) Calculation for HL SubNetwork Links
            #############################################################

            # loop over each link in the HL4 SubNetwork
            for link_idx in range(len(HL_subnet_links)):

                # loop over each Frequency Plan (FP) counter (assumed 20 possible FPs per link)
                for FP_counter in range(self.FP_max_num):

                    # initialize flag to check if an FP has been counted for this link in this iteration
                    FP_flag = 0
    
                    # check for L-Band Link Utilization (indices self.supC_band_separation_idx to end in LSP_array)
                    if any(self.LSP_array[self.supC_band_separation_idx:, HL_links_indices[link_idx], FP_counter]) != 0:

                        # increment the L-Band link count for the current year
                        self.num_link_LBand_annual[year - 1, HL_links_indices[link_idx]] += 1

                        # if no FP has been counted yet for this link:
                        if FP_flag == 0:
                            
                            # Set flag indicating an FP was used for this link
                            FP_flag = 1
                            
                            # update the FP usage count for the link in the current year
                            self.Year_FP_new[year - 1, link_idx] += 1
                
                            # add to the total effective FP with a weight factor (multiplied by 2 for bidirectional consideration)
                            self.Total_effective_FP_new_annual[year - 1] += 2 * self.network.weights_array[HL_links_indices[link_idx]]

                    # check for SuperC-Band Link Utilization (indices self.C_band_separation_idx to self.supC_band_separation_idx in LSP_array)
                    if any(self.LSP_array[self.C_band_separation_idx:self.supC_band_separation_idx, HL_links_indices[link_idx], FP_counter]) != 0:

                        # increment the SuperC-Band link count for the current year
                        self.num_link_SupCBand_annual[year - 1, HL_links_indices[link_idx]] += 1

                        # if no FP has been counted yet for this link:
                        if FP_flag == 0:
                            
                            # Set flag indicating an FP was used for this link
                            FP_flag = 1
                            
                            # update the FP usage count for the link in the current year
                            self.Year_FP_new[year - 1, link_idx] += 1
                
                            # add to the total effective FP with a weight factor (multiplied by 2 for bidirectional consideration)
                            self.Total_effective_FP_new_annual[year - 1] += 2 * self.network.weights_array[HL_links_indices[link_idx]]
                            
                    # check for C-Band Link Utilization (indices 0 to self.C_band_separation_idx in LSP_array)
                    if any(self.LSP_array[:self.C_band_separation_idx, HL_links_indices[link_idx], FP_counter]) != 0:

                        # increment the C-Band link count for the current year
                        self.num_link_CBand_annual[year - 1, HL_links_indices[link_idx]] += 1

                        # if no FP has been counted yet for this link:
                        if FP_flag == 0:
                            
                            # Set flag indicating an FP was used for this link
                            FP_flag = 1
                            
                            # update the FP usage count for the link in the current year
                            self.Year_FP_new[year - 1, link_idx] += 1
                
                            # add to the total effective FP with a weight factor (multiplied by 2 for bidirectional consideration)
                            self.Total_effective_FP_new_annual[year - 1] += 2 * self.network.weights_array[HL_links_indices[link_idx]]

            # Find the links with no use and set their fiber pair to one (at least one FP is exist in each link)
            not_used_links = np.where(self.Year_FP_new[year - 1, :] == 0)
            self.Year_FP_new[year - 1, not_used_links] += 1

            # Initialize GSNR arrays for save GSNR of BVTs
            GSNR_BVT_this_year_primary = []
            GSNR_BVT_this_year_secondary = []
            GSNR_BVT_this_year_all = []
            
            # Extract GSNR values of BVTs in this year:
            for BVT_info_list in self.BVT_establishment_info:
                for BVT_info in BVT_info_list:
                    if BVT_info[3] == 1: # BVT for primary path
                        GSNR_BVT_this_year_primary.append(BVT_info[5])
                        GSNR_BVT_this_year_all.append(BVT_info[5])
                    if BVT_info[3] == -1: # BVT for secondary path
                        GSNR_BVT_this_year_secondary.append(BVT_info[5])
                        GSNR_BVT_this_year_all.append(BVT_info[5])
            
             # Store snapshot of GSNR values in this year           
            self.GSNR_BVT_primary_annual[year - 1] = GSNR_BVT_this_year_primary
            self.GSNR_BVT_secondary_annual[year - 1] = GSNR_BVT_this_year_secondary
            self.GSNR_BVT_all_annual[year - 1] = GSNR_BVT_this_year_all

        # Updating Node degress based on fiber-pairs
        self._update_hl_node_degrees(hierarchy_level = hierarchy_level,
                                     Year_FP = self.Year_FP)
        
        # calculate cimulative BVT usage and license activation
        self._calculate_BVT_usage()
        
        # save all simulation results
        self._save_network_results(hierarchy_level = hierarchy_level, 
                                   minimum_hierarchy_level = minimum_level, 
                                   result_directory = result_directory)


