from typing import List, Tuple
import numpy as np
import pandas as pd
from .network import Network
import matplotlib.pyplot as plt
from itertools import accumulate

class analyse_result:
    """"
    A class for analyzing and managing results from hierarchical optical network simulations.

    This class supports loading network planning results for different hierarchy levels,
    including link-level and transceiver-level (BVT) metrics.

    Attributes:
        network (Network): An instance of the Network class defining topology and metadata.
        period_time (int): Duration (in years or time units) for planning or simulation.
        processing_level_list (List[int]): List of hierarchy levels to be analyzed (e.g., [2, 3, 4]).
        results_directory (Path): Directory containing result .npz files for each hierarchy level.
        save_directory (Path): Directory where output plots will save in.
        link_data (Dict[str, np.lib.npyio.NpzFile]): Dictionary containing link-level data for each HL.
        bvt_data (Dict[str, np.lib.npyio.NpzFile]): Dictionary containing transceiver (BVT) data for each HL.
    """
    
    def __init__(self,
                 network_instance: Network,
                 period_time: int, 
                 processing_level_list: List, 
                 results_directory, 
                 save_directory):
        """
        Initialize the AnalyseResult object.

        Args:
        --------
            network_instance (Network): 
                The optical network structure.
            period_time (int): 
                Simulation or planning time period (e.g., 10 years).
            processing_level_list (List[int]): 
                Hierarchy levels to process (e.g., [2, 3, 4]).
            results_directory(Path): 
                Path to the directory where .npz result files are stored.
            save_directory (Path): 
                Path to the directory where output plots will save in.

        Example:
        --------
        >>> from sixgman.core.optical_result_analyzer import analyse_result
        >>> analyser = analyse_result(
        ...     net, # Network instance
        ...     10, # Planning period time in years
        ...     processing_level_list, # List of hierarchy levels to analyze
        ...     results_dir # Directory where results are saved
        ... )
        """
        self.network = network_instance
        self.period_time = period_time
        self.processing_level_list = processing_level_list
        self.results_directory = results_directory
        self.save_directory = save_directory
        self.link_data = {}
        self.bvt_data = {}
        
        save_directory.mkdir(exist_ok=True)
        
    def _load_data(self):

        """
        Load link and BVT result data for each hierarchy level.

        This method reads .npz files generated for each HL level and stores the parsed
        data in the class attributes `link_data`, `bvt_data` and `GSNR_data`.

        Raises:
            IOError: If any expected file is missing or unreadable.
        """
        link_data = {}
        bvt_data = {}
        GSNR_data = {}
        for hierarchy_level in self.processing_level_list:

            # Construct expected file paths
            link_data_path = self.results_directory / f"{self.network.topology_name}_HL{hierarchy_level}_link_info.npz"
            bvt_data_path = self.results_directory / f"{self.network.topology_name}_HL{hierarchy_level}_bvt_info.npz"
            GSNR_data_path = self.results_directory / f"{self.network.topology_name}_HL{hierarchy_level}_path_GSNR_info.npz"
            
            # Load link information
            try:
                        
                hl_link_data = np.load(link_data_path)
                link_data[f"HL{hierarchy_level}"] = hl_link_data
                self.link_data = link_data

            except Exception as e:
                raise IOError(f"Failed to load data from {link_data_path}: {str(e)}")

            # Load BVT (transceiver) information
            try:
                hl_BVT_data = np.load(bvt_data_path)
                bvt_data[f"HL{hierarchy_level}"] = hl_BVT_data  
                self.bvt_data = bvt_data

            except Exception as e:
                raise IOError(f"Failed to load data from {bvt_data_path}: {str(e)}")
            
            # Load path GSNR information
            try:
                path_GSNR_data = np.load(GSNR_data_path, allow_pickle = True)
                GSNR_data[f"HL{hierarchy_level}"] = path_GSNR_data
                self.GSNR_data = GSNR_data

            except Exception as e:
                raise IOError(f"Failed to load data from {GSNR_data_path}: {str(e)}")
        

        
            
    def plot_link_state(self, 
                        minimum_hierarchy_level: int, 
                        splitter: List,
                        save_flag: int, 
                        save_suffix: str = "",
                        flag_plot: int = 1):
        """
        Plot or return the evolution of link states (Fiber Pairs numbers) across all hierarchy levels over time.

        This function creates a heatmap-style visualization of link states (FP numbers) over a multi-year period,
        where each row corresponds to a specific link and each column to a simulation year. Vertical dashed lines
        separate different hierarchical levels using the provided splitter.

        Args: 
        --------
            splitter (List): 
                A list indicating how many links are in each hierarchy level (used for dashed separators). 
            save_flag (int): 
                If 1, the plot will be saved to disk. 
            save_suffix (str, optional): 
                Custom suffix for the saved file name. Default is "". 
            flag_plot (int, optional): 
                If 1, show the plot; otherwise, return the data. Default is 1. 

        Output: 
        --------
            np.ndarray: 
                If flag_plot == 0, returns the link state matrix of shape (years, total_links). Each element corresponds to the FP number assigned to a specific link in a given year.
                        
        Notes: 
        --------
            - When save_flag = 1, the figure is saved in the output directory with the filename pattern {topology_name}_Link_State{save_suffix}.png 
            - The visualization allows monitoring the temporal evolution of FP allocation and link utilization across all hierarchical levels. 

        Example:
        ---------
        >>> analyser.plot_link_state(
        ...     splitter, # List of integers indicating the number of links per hierarchy level
        ...     save_flag = 0, # If 1, save the plot; if 0, do not save
        ...     ave_suffix = "_NoBypass" # Optional suffix for the saved file name
        ... )
        """
        self._load_data()

        # Initialize an empty array to hold FP numbers across all hierarchy levels and years
        link_state_HL_partisioned = np.empty(shape=(self.period_time, 0))

        # Concatenate link FP number arrays from all hierarchy levels
        for hierarchy_level in self.processing_level_list:

            sub_graph_HL, _ = self.network.compute_hierarchy_subgraph(hierarchy_level, minimum_hierarchy_level)
            HL_subnet_links = np.array(list(sub_graph_HL.edges(data = 'weight')))
            idx_sort = np.argsort(HL_subnet_links[:, 2])  

            HL_FPNum_sorted = self.link_data[f"HL{hierarchy_level}"]['HL_FPNum'][:, idx_sort]


            link_state_HL_partisioned = np.hstack((
                link_state_HL_partisioned,
                HL_FPNum_sorted
            ))

        year = np.arange(1, self.period_time + 1)  # Year range for x-axis

        if flag_plot == 1:
            # Plot the link state profile
            plt.figure(figsize=(7, 5))
            plt.title("ALL HLs: link state profile")

            # Show the heatmap image of link state transitions
            plt.imshow(link_state_HL_partisioned.T,
                        aspect='auto',
                        interpolation='none',
                        origin='upper',
                        extent=[1, self.period_time, link_state_HL_partisioned.shape[1], 1])

            plt.xlabel("Year")
            plt.ylabel("Link index")

            # Set color limits (0 = unused, 1–3 = different link states)
            plt.clim(0, 3)
            plt.colorbar(label='State')

            # Define x-axis limits and ticks
            plt.xlim(1, self.period_time)
            plt.xticks(np.arange(1, self.period_time + 1))  # Adjust as needed if period_time != 10

            # Add grid to the plot
            plt.grid(True)

            # Convert splitter into cumulative index positions to separate HL groups
            splitter_converted = list(accumulate(splitter))

            # Plot dashed lines between hierarchical levels
            for i in range(len(splitter_converted)):
                plt.plot(year,
                        splitter_converted[i] * np.ones_like(year),
                        'k--',
                        linewidth=1)
            
            plt.ylim(splitter_converted[-1])
            plt.yticks(np.arange(0, splitter_converted[-1], 30))

            plt.tight_layout()

            # Save the plot if requested
            if save_flag:
                plt.savefig(self.save_directory / f"{self.network.topology_name}_Link_State{save_suffix}.png",
                            dpi=300, bbox_inches='tight')

            plt.show()

        else:
            # Return the raw matrix if no plot is requested
            return link_state_HL_partisioned

    def plot_FP_usage(self,
                      save_flag: int, 
                      save_suffix: str = "",
                      flag_plot: int = 1):
        """
        Plot and optionally save the Fiber Pair (FP) usage over time across all hierarchy levels.

        This function visualizes:
            - The cumulative FP usage in kilometers (left y-axis, log scale)
            - The total number of FPs (right y-axis, linear scale)

        Args:
        --------
            save_flag (int): 
                If 1, saves the plot to the result directory.
            save_suffix (str): 
                Optional suffix to append to the saved filename.
            flag_plot (int): 
                If 1, display the plot; if 0, skip plotting.

        Output:
        --------
            None
            
        Notes:
        --------
            - The cumulative FP usage represents the total deployed fiber length across all active links and years. 
		    - When save_flag = 1, the figure is stored as {topology_name}_FP_Usage{save_suffix}.png in the result directory. 

        Example:
        --------
        >>> analyser.plot_FP_usage(
        ...     save_flag = 0, # If 1, save the plot; if 0, do not save
        ...     save_suffix = "_NoBypass" # Optional suffix for the saved file name
        ... )

        """
        self._load_data()  # Load link data for all hierarchy levels
        year = np.arange(1, self.period_time + 1)  # e.g., years 1 to 10

        if flag_plot == 1:
            fig, ax1 = plt.subplots(figsize=(7, 5))
            fig.suptitle("Total FP Usage [km] and [number of FP]")
            ax2 = ax1.twinx()  # Create secondary y-axis for FP count

        default_markers = ['d', 'p', 's', 'o', '>', 'h', '*', 'x']  # Marker styles per HL

        km_total = 0
        fp_total = 0
        for hierarchy_level in self.processing_level_list:
            km = self.link_data[f"HL{hierarchy_level}"]['Total_effective_FP_new_annual']
            fp = np.sum(self.link_data[f"HL{hierarchy_level}"]['HL_FPNum'], axis=1)

            if flag_plot == 1:
                # Left Y-axis: FP usage in km (log scale)
                ax1.semilogy(year, km,
                            f"b-{default_markers[hierarchy_level]}",
                            label=f"HL{hierarchy_level}s [km]", linewidth=1.5)
                # Right Y-axis: FP count (linear scale)
                ax2.plot(year, fp,
                        f"r-.{default_markers[hierarchy_level]}",
                        label=f"HL{hierarchy_level}s [#]", linewidth=1.5)

            km_total += km
            fp_total += fp

        if flag_plot == 1:
            # Plot total across all HLs
            ax1.semilogy(year, km_total, "b-+", label="Total [km]", linewidth=1.5)
            ax1.set_ylabel("Cumulative Fiber Pair Usage [Km]", color="blue")
            ax1.set_xlabel("Year")
            ax1.tick_params(axis="y", labelcolor="blue")

            ax2.plot(year, fp_total, "r-+", label="Total [#]", linewidth=1.5)
            ax2.set_ylabel("Cumulative Number of Used Fiber Pairs", color="red")
            ax2.tick_params(axis="y", labelcolor="red")
            # ax2.set_ylim(0, 250)
            # ax2.set_yticks(np.arange(0, 251, 50))

            # Add grid and combined legend
            ax1.grid(True)
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc="upper left")
            plt.tight_layout()

            if save_flag:
                plt.savefig(self.save_directory / f"{self.network.topology_name}_FP_Usage{save_suffix}.png",
                            dpi=300, bbox_inches='tight')

            plt.show()

    def plot_FP_degree(self,
                       save_flag: int, 
                       save_suffix: str = "",
                       flag_plot: int = 1):
        """
        Plot and optionally save the cumulative Fiber Pair (FP) usage and Band degree types.

        This function visualizes:
            - Total FP usage in kilometers (left Y-axis, log scale)
            - Nodal degrees for C-band, SuperC-band, and L-band (right Y-axis, linear scale)

        Args:
        ---------
            save_flag (int): 
                If 1, saves the plot as an image to the save_directory.
            save_suffix (str): 
                Optional suffix to append to the saved filename.
            flag_plot (int): 
                If 1, plot will be displayed. If 0, no plot is shown.

        Output:
        ---------
            None
            
        Notes:
        ---------
            - FP usage represents the total deployed fiber length in kilometers, summed across all links and years. 
	        - Nodal degree metrics indicate the number of active fiber-pair connections per node in each spectral band. 
	        - When save_flag = 1, the plot is stored in the results directory with filename pattern {topology_name}_FP_Degree{save_suffix}.png}. 

        Example:
        --------
        >>> analyser.plot_FP_degree(
        ...     save_flag = 0, # If 1, save the plot; if 0, do not save
        ...     save_suffix = "_NoBypass" # Optional suffix for the saved file name
        ... )

        """
        self._load_data()
        year = np.arange(1, self.period_time + 1)

        if flag_plot == 1:
            fig, ax1 = plt.subplots(figsize=(7, 5))
            fig.suptitle("FP [km] and Degree")
            ax2 = ax1.twinx()  # Secondary axis for nodal degree plots

        default_markers = ['d', 'p', 's', 'o', '>', 'h', '*', 'x']

        km_total = 0
        deg_c = 0
        deg_superc = 0
        deg_l = 0

        for hierarchy_level in self.processing_level_list:
            km = self.link_data[f"HL{hierarchy_level}"]['Total_effective_FP_new_annual']
            if flag_plot == 1:
                ax1.semilogy(year, km,
                            f"b-{default_markers[hierarchy_level]}",
                            label=f"HL{hierarchy_level}s [km]", linewidth=1.5)
            km_total += km

        deg_c = self.link_data[f"HL{self.processing_level_list[-1]}"]['HL_CDegree_Domain'].sum(axis = 1)
        deg_superc = self.link_data[f"HL{self.processing_level_list[-1]}"]['HL_SuperCDegree_Domain'].sum(axis = 1)
        deg_l = self.link_data[f"HL{self.processing_level_list[-1]}"]['HL_LDegree_Domain'].sum(axis = 1)

        # Store for external use if needed
        self.deg_pure_c = deg_c - deg_superc - deg_l
        self.deg_c = deg_c
        self.deg_superc = deg_superc
        self.deg_l = deg_l

        if flag_plot == 1:
            ax1.semilogy(year, km_total,
                        "b-+",
                        label="Total [km]", linewidth=1.5)
            ax1.set_ylabel("Cumulative Fiber Pair Usage [Km]", color="blue")
            ax1.set_xlabel("Year")
            ax1.tick_params(axis="y", labelcolor="blue")

            # Right Y-axis: Degrees
            ax2.set_ylabel("Cumulative Band Degree Number", color="red")
            ax2.plot(year, deg_c, 'r-.h', label='C-Band-Degree [#]', linewidth=1.5)
            ax2.plot(year, deg_superc, 'r-.*', label='SupC-Band-Degree [#]', linewidth=1.5)
            ax2.plot(year, deg_l, 'r-.s', label='L-Band-Degree [#]', linewidth=1.5)
            ax2.tick_params(axis="y", labelcolor="red")
            # ax2.set_ylim(0, 800)
            # ax2.set_yticks(np.arange(0, 801, 100))

            # Combined legend and final layout
            ax1.grid(True)
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc="upper left")
            plt.tight_layout()

            if save_flag:
                plt.savefig(self.save_directory / f"{self.network.topology_name}_FP_Degree{save_suffix}.png",
                            dpi=300, bbox_inches='tight')

            plt.show()


    def plot_bvt_license(self, 
                         save_flag: int, 
                         save_suffix: str = "",
                         flag_plot: int = 1):
        """
        Plot and optionally save the cumulative BVT usage and 100G license allocation over time.

        This function visualizes:
            - The cumulative count of BVTs (Bandwidth Variable Transponders) for each band: 
                C-Band, SuperC-Band, and L-Band (left Y-axis, log scale). 
            - The corresponding allocation of 100G licenses proportionally distributed across 
                the bands (right Y-axis, linear scale). 

        Args:
        ---------
            save_flag (int): 
                If 1, saves the plot as an image to the save_directory.
            save_suffix (str): 
                Optional suffix for the saved image filename.
            flag_plot (int): 
                If 1, displays the plot; if 0, skips plotting.

        Output:
        ---------
            None
            
        Notes:
        ---------
            - The plotted data reflects cumulative deployment across simulation years, combining all hierarchy levels. 
	        - 100G license counts are derived proportionally from BVT allocations, assuming four 100G channels per BVT. 
	        - When save_flag = 1, the plot is saved in the results directory under the filename pattern {topology_name}_BVT_License{save_suffix}.png. 

        Example:
        --------
        >>> analyser.plot_bvt_license(
        ...     save_flag = 0, # If 1, save the plot; if 0, do not save
        ...     save_suffix = "_NoBypass" # Optional suffix for the saved file name
        ... )

        """
        self._load_data()
        year = np.arange(1, self.period_time + 1)

        if flag_plot == 1:
            # Initialize figure and axis
            fig, ax1 = plt.subplots(figsize=(7, 5))
            fig.suptitle("Total BVT and 100G-License")
            ax2 = ax1.twinx()  # Right-side Y-axis

        # Initialize cumulative BVT and license counters
        All_BVT_CBand = 0
        All_BVT_SuperC = 0
        All_BVT_L = 0
        All_license_CBand = 0
        All_license_SuperC = 0
        All_license_L = 0
        Total_License = 0

        # Accumulate values across all processing levels
        for hierarchy_level in self.processing_level_list:
            All_BVT_CBand += self.bvt_data[f"HL{hierarchy_level}"]['HL_BVTNum_CBand']
            All_BVT_SuperC += self.bvt_data[f"HL{hierarchy_level}"]['HL_BVTNum_SuperCBand']
            All_BVT_L += self.bvt_data[f"HL{hierarchy_level}"]['HL_BVTNum_LBand']
            All_license_CBand += self.bvt_data[f"HL{hierarchy_level}"]['HL_CBand_license']
            All_license_SuperC += self.bvt_data[f"HL{hierarchy_level}"]['HL_SuperCBand_license']
            All_license_L += self.bvt_data[f"HL{hierarchy_level}"]['HL_LBand_license']
            Total_License += self.bvt_data[f"HL{hierarchy_level}"]['HL_All_100G_lincense']

        # Save accumulated results for external access
        self.All_BVT_CBand = All_BVT_CBand
        self.All_BVT_SuperC = All_BVT_SuperC
        self.All_BVT_L = All_BVT_L

        # Calculate total BVTs and proportional 100G license allocation
        Total_BVT = All_BVT_CBand + All_BVT_SuperC + All_BVT_L
        self.CBand_100G_License = All_license_CBand.sum(axis = 1)
        self.SupCBand_100G_License = All_license_SuperC.sum(axis = 1)
        self.LBand_100G_License = All_license_L.sum(axis = 1)
        Total_100G_License = (
            self.CBand_100G_License +
            self.SupCBand_100G_License +
            self.LBand_100G_License
        )

        if flag_plot == 1:
            # --- Plot BVT Counts on Left Y-axis ---
            ax1.set_xlabel("Year")
            ax1.set_ylabel("Cumulative BVT Number", color='blue')
            ax1.semilogy(year, All_BVT_CBand, 'b->', label='C-Band-BVT[#]', linewidth=1.5)
            ax1.semilogy(year, All_BVT_SuperC, 'b-o', label='SupC-Band-BVT[#]', linewidth=1.5)
            ax1.semilogy(year, All_BVT_L, 'b-s', label='L-Band-BVT[#]', linewidth=1.5)
            ax1.semilogy(year, Total_BVT, 'b-+', label='Total-BVT[#]', linewidth=1.5)
            ax1.tick_params(axis='y', labelcolor='blue')

            # --- Plot 100G Licenses on Right Y-axis ---
            ax2.set_ylabel("Cumulative 100G - License Number", color='red')
            ax2.plot(year, self.CBand_100G_License.cumsum(), 'r-.>', label='C-Band-100GL[#]', linewidth=1.5)
            ax2.plot(year, self.SupCBand_100G_License.cumsum(), 'r-.o', label='SupC-Band-100GL', linewidth=1.5)
            ax2.plot(year, self.LBand_100G_License.cumsum(), 'r-.s', label='L-Band-100GL', linewidth=1.5)
            ax2.plot(year, Total_100G_License.cumsum(), 'r-.+', label='Total-100GL', linewidth=1.5)
            ax2.tick_params(axis='y', labelcolor='red')

            # --- Finalize plot ---
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc='upper left')
            ax1.grid(True)
            plt.tight_layout()

            # Save if requested
            if save_flag:
                plt.savefig(self.save_directory / f"{self.network.topology_name}_BVT_License{save_suffix}.png",
                            dpi=300, bbox_inches='tight')
            plt.show()

            
    def calc_cost(self, 
                  save_flag: int,
                  save_suffix: str = "", 
                  C_100GL_First: float = 1,
                  C_100G_Added: float = 0.333, 
                  C_MCS: float = 0.7,
                  C_RoB: float = 1.9,
                  C_IRU: float = 0.5):
        """
        Compute OPEX and CAPEX values for network deployment over time.

        This function evaluates capital and operational expenditures based on 
        nodal degree evolution and bandwidth resource demands (e.g., BVTs, licenses).

        Args:
        ---------
            save_flag (int): 
                Whether to save the output CSV file (1) or not (0).
            save_suffix (str): 
                Optional suffix for the saved file name.
            C_100GL_First (float): 
                Unit cost of the first activated 100G license [default = 1].
            C_100G_Added (float):
                Unit cost of the following activated 100G licenses after the first one [default = 0.333].
            C_MCS (float): 
                Unit cost of a multi-cast switch [default = 0.7].
            C_RoB (float): 
                Unit cost of ROADM on the Blade [default = 1.9].
            C_IRU (float): 
                Cost per km of IRU fiber pair usage [default = 0.5].

        Output:
        ---------
            dict: A dataframe containing columns for OPEX and CAPEX components for different years.
            
        Notes:
        ---------
            - The resulting cost dataframe is used for techno-economic analyses of hierarchical optical networks.
	        - When save_flag = 1, results are stored as a CSV file named {topology_name}_cost_analyse{save_suffix}.csv in the result directory.

        Example:
        ---------
        >>> analyser.calc_cost(
        ...     save_flag = 0, # If 1, save the results; if 0, do not save
        ...     save_suffix = "_NoBypass" # Optional suffix for the saved file name
        ... ) # Returns a dictionary with OPEX and CAPEX components

        """

        cost_dict = {}

        # --- Load data and compute usage from prior functions ---
        self._load_data()
        self.plot_FP_degree(flag_plot=0, save_flag=0)
        self.plot_bvt_license(flag_plot=0, save_flag=0)

        # --- Aggregate all Fiber Pair usage across hierarchy levels ---
        All_FP_km = 0
        for hierarchy_level in self.processing_level_list:
            All_FP_km += self.link_data[f"HL{hierarchy_level}"]['Total_effective_FP_new_annual']

        # --- Compute Operational Expenditure (OPEX) ---
        OPEX = C_IRU * All_FP_km
        cost_dict['OPEX'] = OPEX

        # --- Initialize CAPEX components for each technology and band ---
        Capex_RoB_C = np.zeros(self.period_time)
        Capex_RoB_SupC = np.zeros(self.period_time)
        Capex_RoB_L = np.zeros(self.period_time)

        Capex_MCS_C = np.zeros(self.period_time)
        Capex_MCS_SupC = np.zeros(self.period_time)
        Capex_MCS_L = np.zeros(self.period_time)

        Capex_100GL_Cband = np.zeros(self.period_time)
        Capex_100GL_SupCBand = np.zeros(self.period_time)
        Capex_100GL_LBand = np.zeros(self.period_time)

        # --- Loop over time period to compute annual CAPEX ---
        for y in range(self.period_time):
            if y > 0:
                # Compute annual increment for RoB cost
                Capex_RoB_C[y] = (self.deg_pure_c[y] - self.deg_pure_c[y - 1]) * C_RoB
                Capex_RoB_SupC[y] = (self.deg_superc[y] - self.deg_superc[y - 1]) * C_RoB * (1 + 0.1 * (1 - 0.1)**(y + 1))
                Capex_RoB_L[y] = (self.deg_l[y] - self.deg_l[y - 1]) * C_RoB * (1 + 0.2 * (1 - 0.1)**(y + 1))

                # Compute annual increment for MCS cost (per 16 BVT units)
                Capex_MCS_C[y] = ((self.All_BVT_CBand[y] - self.All_BVT_CBand[y - 1]) / 16) * C_MCS
                Capex_MCS_SupC[y] = ((self.All_BVT_SuperC[y] - self.All_BVT_SuperC[y - 1]) / 16) * C_MCS * (1 + 0.1 * (1 - 0.1)**(y + 1))
                Capex_MCS_L[y] = ((self.All_BVT_L[y] - self.All_BVT_L[y - 1]) / 16) * C_MCS * (1 + 0.2 * (1 - 0.1)**(y + 1))

                # Compute annual increment for 100G License cost
                Added_BVT_annual_CBand = self.All_BVT_CBand[y] - self.All_BVT_CBand[y - 1]
                Added_BVT_annual_supCBand = self.All_BVT_SuperC[y] - self.All_BVT_SuperC[y - 1]
                Added_BVT_annual_LBand = self.All_BVT_L[y] - self.All_BVT_L[y - 1]
                
                Capex_100GL_Cband[y] += Added_BVT_annual_CBand * C_100GL_First
                Capex_100GL_Cband[y] += (self.CBand_100G_License[y] - Added_BVT_annual_CBand) * C_100G_Added
                
                Capex_100GL_SupCBand[y] += Added_BVT_annual_supCBand * C_100GL_First * (1 + 0.1 * (1 - 0.1)**(y + 1))
                Capex_100GL_SupCBand[y] += (self.SupCBand_100G_License[y] - Added_BVT_annual_supCBand) * C_100G_Added * (1 + 0.1 * (1 - 0.1)**(y + 1))
                
                Capex_100GL_LBand[y] += Added_BVT_annual_LBand * C_100GL_First * (1 + 0.2 * (1 - 0.1)**(y + 1))
                Capex_100GL_LBand[y] += (self.LBand_100G_License[y] - Added_BVT_annual_LBand) * C_100G_Added * (1 + 0.2 * (1 - 0.1)**(y + 1))
            else:
                # Initial year costs
                Capex_RoB_C[y] = self.deg_pure_c[y] * C_RoB
                Capex_RoB_SupC[y] = self.deg_superc[y] * C_RoB * 1.1
                Capex_RoB_L[y] = self.deg_l[y] * C_RoB * 1.2

                Capex_MCS_C[y] = self.All_BVT_CBand[y] / 16 * C_MCS
                Capex_MCS_SupC[y] = self.All_BVT_SuperC[y] / 16 * C_MCS * 1.1
                Capex_MCS_L[y] = self.All_BVT_L[y] / 16 * C_MCS * 1.2

                Capex_100GL_Cband[y] += self.All_BVT_CBand[y] * C_100GL_First
                Capex_100GL_Cband[y] += (self.CBand_100G_License[y] - self.All_BVT_CBand[y]) * C_100G_Added
                
                Capex_100GL_SupCBand[y] += self.All_BVT_SuperC[y] * C_100GL_First * 1.1
                Capex_100GL_SupCBand[y] += (self.SupCBand_100G_License[y] - self.All_BVT_SuperC[y]) * C_100G_Added * 1.1
                
                Capex_100GL_LBand[y] += self.All_BVT_L[y] * C_100GL_First * 1.2
                Capex_100GL_LBand[y] = (self.LBand_100G_License[y] - self.All_BVT_L[y]) * C_100G_Added * 1.2

        # --- Aggregate CAPEX components ---
        Capex_RoB = Capex_RoB_C + Capex_RoB_SupC + Capex_RoB_L
        Capex_MCS = Capex_MCS_C + Capex_MCS_SupC + Capex_MCS_L
        Capex_100GL = Capex_100GL_Cband + Capex_100GL_SupCBand + Capex_100GL_LBand
        CAPEX = Capex_RoB + Capex_MCS + Capex_100GL

        cost_dict['Capex_RoB'] = Capex_RoB
        cost_dict['Capex_MCS'] = Capex_MCS
        cost_dict['Capex_100GL'] = Capex_100GL
        cost_dict['CAPEX'] = CAPEX

        # --- Convert results to DataFrame and optionally save ---
        cost_df = pd.DataFrame(cost_dict)

        if save_flag:
            cost_df.to_csv(
                self.save_directory / f"{self.network.topology_name}_cost_analyse{save_suffix}.csv", 
                index=False
            )

        return cost_df
    
    def find_base_router_capacity(self, 
                                  node_IDx: int, 
                                  node_capacity_profile_annual: np.ndarray, 
                                  reference_year: int,
                                  reference_router_capacities: np.ndarray
                                  ):
        
        """
            Select the base capacity of IP router based on a traffic in a reference year.
    
            Args:
            ---------
                node_IDx (int): 
                    IDx of the node that we want to find the base router for.
                node_capacity_profile_annual (np.ndarray): 
                    Traffic profile of nodes in the network 
                reference_year (int): 
                    The reference year that select the base capacity of IP router based on.
                reference_router_capacities (np.ndarray):
                    Array of different router types that can be deployed in the network.
    
            Output:
            ---------
                float: Choosen IP router capacity for the node
                    
            Example:
            ---------
            >>> analyser.find_base_router_capacity(
            ...     node_IDx = 39, # Calc the base IP router capacity for node 30
            ...     node_capacity_profile_annual = node_capacity_profile_annual,
                    reference_year = 5, # the reference year for router selection is year 5
                    reference_router_capacities = np.array([400, 800, 1600, 3200, 6400, 12800, 25600])
            ... ) # Returns the base IP router for node 30
    
        """
            
        ref_traffic = node_capacity_profile_annual[reference_year - 1, node_IDx]
        idx = np.searchsorted(reference_router_capacities, ref_traffic, side='left')
        idx = idx if idx < len(reference_router_capacities) else len(reference_router_capacities) - 1
        chosen_cap = reference_router_capacities[idx]
        return chosen_cap
        
    def compute_electrical_latency(self, 
                                    total_traffic_IP_node_Gbps: int, 
                                    reference_capacity_Gbps: int,
                                    Avg_Bytes_Packet: int = 1019, # in terms of btyes
                                    packet_spike_variance: float = 1.3,
                                    Pth_percentile: float = 0.99, 
                                    processing_delay: float = 1, # in us, 
                                    ):
        
        """
        Computes the expected electrical latency and P-th percentile delay bound for an IP node.

        This function estimates the total delay (queuing + service time) using queuing theory 
        principles. If the incoming traffic exceeds the reference capacity, it assumes a 
        load-balancing scenario where traffic is evenly split between two identical routers.

        Args:
        ---------
            total_traffic_IP_node_Gbps (int): 
                Total incoming traffic to the IP node in Gbps.
            reference_capacity_Gbps (int): 
                The processing capacity of a single router in Gbps.
            Avg_Bytes_Packet (int, optional): 
                Average size of an incoming packet in bytes. 
                Defaults to 1019.
            packet_spike_variance (float, optional): 
                Variance multiplier used to account for bursty traffic in the expected queuing delay calculation. Defaults to 1.3.
            Pth_percentile (float, optional): 
                The target probability threshold for the worst-case delay bound (e.g., 0.99 for 99th percentile). Defaults to 0.99.
            processing_delay (float, optional): 
                Base processing delay in microseconds (µs). Defaults to 1. (Note: Currently unused in the mathematical calculation).

        Output:
        ---------
            float: The P-th percentile worst-case delay bound (D_p) in nanoseconds.

        Example:
        ---------
            >>> node = NetworkNode()
            >>> delay = node.compute_electrical_latency(
            ...     total_traffic_IP_node_Gbps=800,
            ...     reference_capacity_Gbps=1000
            ... )
            >>> print(f"{delay:.2f}")
            350.00
        """
        
        if total_traffic_IP_node_Gbps < reference_capacity_Gbps:
            utilization_factor = total_traffic_IP_node_Gbps / reference_capacity_Gbps
        else: # apply load balancing, divide the traffic between two routers in the same capacities
            utilization_factor = 0.5 * total_traffic_IP_node_Gbps / reference_capacity_Gbps
        
        # Calculate service time
        E_X = Avg_Bytes_Packet * 8 / reference_capacity_Gbps # in terms of ns
        
        # Average queuing delay
        E_W_q = E_X * (utilization_factor / (1 - utilization_factor)) * ((1 + packet_spike_variance) / 2)
        
        E_D = E_W_q + E_X
        std_delay = E_D # in the exponentioal dist the average and standard deviation is equal
        D_p = E_D + np.sqrt((4 / 9) * (1 / (1 - Pth_percentile))) * std_delay
            
        return D_p * 1e-3 + processing_delay
    
    def compute_E2E_path_latency(self, 
                                    node: int, 
                                    latency_core_array_SA_global: np.ndarray, 
                                    destination_core_array_SA_global: np.ndarray,
                                    latency_core_array_co_global: np.ndarray, 
                                    destination_core_array_co_global: np.ndarray,
                                    processing_level_list: list,
                                    hierarchy_level_src: int, 
                                    node_capacity_profile_annual: np.ndarray, 
                                    node_cap_index_list: list,
                                    reference_year: int, 
                                    reference_router_capacities: np.ndarray, 
                                    year: int, 
                                    Avg_Bytes_Packet: int = 1019, 
                                    packet_spike_variance: float = 1.3,
                                    Pth_percentile: float = 0.99, 
                                    processing_delay: float = 1, 
                                    ):
        """
        Recursively compute all possible end-to-end (E2E) latency paths from a given core node 
        to its higher-level (top-hierarchy) destination nodes.

        This method traces both primary and secondary (dual-home) connectivity paths defined 
        in the input arrays, accumulating the total propagation latency and electrical latency 
        along each possible route. The computation proceeds recursively through hierarchy levels 
        until reaching the top-level nodes.

        Args:
        ---------
            node (int): 
                The index of the current node.
            latency_core_array_SA_global (np.ndarray): 
                Precomputed hop latencies for standalone nodes.
            destination_core_array_SA_global (np.ndarray): 
                Destination mappings for standalone nodes.
            latency_core_array_co_global (np.ndarray): 
                Precomputed hop latencies for central office nodes.
            destination_core_array_co_global (np.ndarray): 
                Destination mappings for central office nodes.
            processing_level_list (list): 
                List of network hierarchy levels (e.g., [4, 3, 2, 1]).
            hierarchy_level_src (int): 
                The hierarchy level of the current source node.
            node_capacity_profile_annual (np.ndarray): 
                Annual traffic profiles for all nodes.
            node_cap_index_list (list): 
                Indices mapping hierarchy levels to traffic profiles.
            reference_year (int): 
                The baseline year for capacity planning.
            reference_router_capacities (np.ndarray): 
                Available router capacities to match against.
            year (int): 
                The specific year for which the latency is being calculated.
            Avg_Bytes_Packet (int, optional): 
                Average packet size in bytes. Defaults to 1019.
            packet_spike_variance (float, optional): 
                Variance multiplier for bursty traffic. Defaults to 1.3.
            Pth_percentile (float, optional): 
                Target probability threshold for latency bound. Defaults to 0.99.
            processing_delay (float, optional): 
                Base processing delay in microseconds. Defaults to 1.

        Outputs:
        ---------
            list: A list of tuples, where each tuple contains (path_list, total_latency).
                Example: [([node1, node2, node3], 450.5), ...]

        Example:
        ---------
            >>> # Assuming 'self' is an instantiated network object
            >>> paths = self.compute_E2E_path_latency(
            ...     node=5, latency_core_array_SA_global=lat_SA, destination_core_array_SA_global=dest_SA,
            ...     latency_core_array_co_global=lat_CO, destination_core_array_co_global=dest_CO,
            ...     processing_level_list=[4, 3, 2, 1], hierarchy_level_src=4,
            ...     node_capacity_profile_annual=traffic_profile, node_cap_index_list=[0, 1, 2, 3],
            ...     reference_year=2024, reference_router_capacities=router_caps, year=2026
            ... )
            >>> print(paths[0])
            ([5, 12, 0], 1250.75)
        """
        hl_src_idx = np.where(np.array(processing_level_list) == hierarchy_level_src)[0][0]
        node_name_in_hl = self.network.all_nodes[node]
        if self.network.get_standalone_hierarchy_level(node_name_in_hl) == hierarchy_level_src:
            lat_paths = latency_core_array_SA_global[hl_src_idx][node]
            dests = destination_core_array_SA_global[hl_src_idx][node]
            if dests == 0:
                dests = ()
        else:
            lat_paths = latency_core_array_co_global[hl_src_idx][node]
            dests = destination_core_array_co_global[hl_src_idx][node]
            if dests == 0:
                dests = ()
            
        node_capacity_profile_annual_hl = node_capacity_profile_annual[:, :, node_cap_index_list[hl_src_idx]]

        results = []
        for i, d in enumerate(dests):
            d = int(d)
            if d != -1: # check for not valid destinations
                hop_latency = float(lat_paths[i])
                base_cap_router = self.find_base_router_capacity(node_IDx = d, 
                                                                node_capacity_profile_annual = node_capacity_profile_annual_hl, 
                                                                reference_year = reference_year, 
                                                                reference_router_capacities = reference_router_capacities)
                
                total_traffic_dest = node_capacity_profile_annual_hl[year - 1, d]
                electrical_latency = self.compute_electrical_latency(total_traffic_IP_node_Gbps = total_traffic_dest, 
                                                                    reference_capacity_Gbps = base_cap_router, 
                                                                    Avg_Bytes_Packet = Avg_Bytes_Packet, 
                                                                    Pth_percentile = Pth_percentile, 
                                                                    packet_spike_variance = packet_spike_variance, 
                                                                    processing_delay = processing_delay)
                

                if hierarchy_level_src != processing_level_list[-1]:
                    # Recursive call
                    subpaths = self.compute_E2E_path_latency(d, latency_core_array_SA_global, destination_core_array_SA_global, latency_core_array_co_global, 
                                                                destination_core_array_co_global, processing_level_list, processing_level_list[hl_src_idx + 1], 
                                                                node_capacity_profile_annual, node_cap_index_list, reference_year, reference_router_capacities, year)
                    for subpath, sublat in subpaths:
                        results.append(([node] + subpath, hop_latency + sublat + electrical_latency))
                else:
                    results.append(([node, d], hop_latency + electrical_latency))
                    
        return results

    def calc_E2E_latency_Total(self, 
                                latency_core_array_SA_global: np.ndarray, 
                                destination_core_array_SA_global: np.ndarray,
                                latency_core_array_co_global: np.ndarray, 
                                destination_core_array_co_global: np.ndarray,
                                processing_level_list: list,
                                save_flag: int, 
                                node_capacity_profile_annual: np.ndarray, 
                                node_cap_index_list: list,
                                reference_year: int, 
                                reference_router_capacities: np.ndarray, 
                                year: int, 
                                Avg_Bytes_Packet: int = 1019, 
                                packet_spike_variance: float = 1.3,
                                Pth_percentile: float = 0.99, 
                                processing_delay: float = 1,
                                save_suffix: str = ""
                                ):
        """
        Compute and optionally save the total end-to-end (E2E) latency from the lowest 
        hierarchy-level nodes up to the top-level core nodes.

        This function iterates over all nodes at the lowest hierarchy level and calls 
        `compute_E2E_path_latency` recursively to determine the cumulative latency of all 
        possible paths. Results can be saved to a `.npz` file.

        Args:
        ---------
            latency_core_array_SA_global (np.ndarray): 
                Precomputed hop latencies for SA nodes.
            destination_core_array_SA_global (np.ndarray): 
                Destination mappings for SA nodes.
            latency_core_array_co_global (np.ndarray): 
                Precomputed hop latencies for CO nodes.
            destination_core_array_co_global (np.ndarray): 
                Destination mappings for CO nodes.
            processing_level_list (list): 
                List of network hierarchy levels (e.g., [4, 3, 2, 1]).
            save_flag (int): 
                1 to save the resulting array to disk, 0 otherwise.
            node_capacity_profile_annual (np.ndarray): 
                Annual traffic profiles for all nodes.
            node_cap_index_list (list): 
                Indices mapping hierarchy levels to traffic profiles.
            reference_year (int): 
                The baseline year for capacity planning.
            reference_router_capacities (np.ndarray): 
                Available router capacities to match against.
            year (int): 
                The specific year for which the latency is being calculated.
            Avg_Bytes_Packet (int, optional): 
                Average packet size in bytes. Defaults to 1019.
            packet_spike_variance (float, optional): 
                Variance multiplier for bursty traffic. Defaults to 1.3.
            Pth_percentile (float, optional): 
                Target probability threshold for latency bound. Defaults to 0.99.
            processing_delay (float, optional): 
                Base processing delay in microseconds. Defaults to 1.
            save_suffix (str, optional): 
                Suffix appended to the saved file name. Defaults to "".

        Outputs:
        ---------
            np.ndarray: A 1D array containing the total latencies of all E2E paths found.

        Example:
        ---------
            >>> # Assuming 'self' is an instantiated network object
            >>> total_latencies = self.calc_E2E_latency_Total(
            ...     latency_core_array_SA_global=lat_SA, destination_core_array_SA_global=dest_SA,
            ...     latency_core_array_co_global=lat_CO, destination_core_array_co_global=dest_CO,
            ...     processing_level_list=[4, 3, 2, 1], save_flag=1,
            ...     node_capacity_profile_annual=traffic_profile, node_cap_index_list=[0, 1, 2, 3],
            ...     reference_year=2024, reference_router_capacities=router_caps, year=2026
            ... )
            >>> print(total_latencies[:3])
            [1250.75 1310.20 1405.00]
        """
        # HLx hierarchy is traversed top-down; 
        minimum_hierarchy_level = processing_level_list[0]
        minimum_HL_nodes = self.network.hierarchical_levels[f'HL{minimum_hierarchy_level}']['standalone']
        
        E2E_latency_list = []
        
        for HL_node_IDx in range(len(np.where(np.isin(self.network.all_nodes, minimum_HL_nodes))[0])):
            HL_node_in_net = np.where(np.isin(self.network.all_nodes, minimum_HL_nodes))[0][HL_node_IDx]
            HL_node = minimum_HL_nodes[HL_node_IDx]
            hierarchy_level_src = self.network.get_standalone_hierarchy_level(HL_node)
            latency_list = self.compute_E2E_path_latency(HL_node_in_net,
                                                        latency_core_array_SA_global,
                                                        destination_core_array_SA_global,
                                                        latency_core_array_co_global,
                                                        destination_core_array_co_global,
                                                        processing_level_list, 
                                                        hierarchy_level_src,
                                                        node_capacity_profile_annual, 
                                                        node_cap_index_list,
                                                        reference_year, 
                                                        reference_router_capacities, 
                                                        year, 
                                                        Avg_Bytes_Packet, 
                                                        packet_spike_variance, 
                                                        Pth_percentile, 
                                                        processing_delay)
                
            for i in range(len(latency_list)):
                _, latency = latency_list[i]                    
                E2E_latency_list.append(latency)
                
        E2E_latency_list = np.array(E2E_latency_list)

        # Optional: Save to disk in compressed format
        if save_flag:
            np.savez_compressed(
                self.save_directory / f"{self.network.topology_name}_latency{save_suffix}.npz",
                latency = E2E_latency_list
            )

        return E2E_latency_list
    
    def compute_E2E_path_distance(self, 
                                    node: int, 
                                    latency_core_array_SA_global: np.ndarray, 
                                    destination_core_array_SA_global: np.ndarray,
                                    latency_core_array_co_global: np.ndarray, 
                                    destination_core_array_co_global: np.ndarray,
                                    processing_level_list: list,
                                    hierarchy_level_src: int):
        """
        Recursively compute all possible end-to-end (E2E) physical distances from a given core node 
        to its higher-level (top-hierarchy) destination nodes.

        This method traces connectivity paths and calculates the distance by converting 
        precomputed hop latencies back into physical distances (assuming a constant of 5 µs/km).
        The computation proceeds recursively through hierarchy levels until reaching 
        the top-level nodes.

        Args:
        ---------
            node (int): 
                The index of the current node.
            latency_core_array_SA_global (np.ndarray): 
                Precomputed hop latencies for SA nodes.
            destination_core_array_SA_global (np.ndarray): 
                Destination mappings for SA nodes.
            latency_core_array_co_global (np.ndarray): 
                Precomputed hop latencies for CO nodes.
            destination_core_array_co_global (np.ndarray): 
                Destination mappings for CO nodes.
            processing_level_list (list): 
                List of network hierarchy levels (e.g., [4, 3, 2, 1]).
            hierarchy_level_src (int): 
                The hierarchy level of the current source node.

        Outputs:
        ---------
            list: A list of tuples, where each tuple contains (path_list, total_distance_km).
                Example: [([node1, node2, node3], 250.5), ...]

        Example:
        ---------
            >>> # Assuming 'self' is an instantiated network object
            >>> distances = self.compute_E2E_path_distance(
            ...     node=5, latency_core_array_SA_global=lat_SA, destination_core_array_SA_global=dest_SA,
            ...     latency_core_array_co_global=lat_CO, destination_core_array_co_global=dest_CO,
            ...     processing_level_list=[4, 3, 2, 1], hierarchy_level_src=4
            ... )
            >>> print(distances[0])
            ([5, 12, 0], 250.15)
        """
        hl_src_idx = np.where(np.array(processing_level_list) == hierarchy_level_src)[0][0]
        if self.network.get_standalone_hierarchy_level(node) == hierarchy_level_src:
            lat_paths = latency_core_array_SA_global[hl_src_idx][node]
            dests = destination_core_array_SA_global[hl_src_idx][node]
            if dests == 0:
                dests = ()
        else:
            lat_paths = latency_core_array_co_global[hl_src_idx][node]
            dests = destination_core_array_co_global[hl_src_idx][node]
            if dests == 0:
                dests = ()
            
        results = []
        for i, d in enumerate(dests):
            d = int(d)
            if d != -1: # check for not valid destinations
                hop_dist = float(lat_paths[i]) / 5
                if hierarchy_level_src != processing_level_list[-1]:
                    # Recursive call
                    subpaths = self.compute_E2E_path_distance(d, latency_core_array_SA_global, destination_core_array_SA_global, latency_core_array_co_global, 
                                                                destination_core_array_co_global, processing_level_list, processing_level_list[hl_src_idx + 1])
                    
                    for subpath, subdist in subpaths:
                        results.append(([node] + subpath, hop_dist + subdist))
                else:
                    results.append(([node, d], hop_dist))
                    
        return results

    def calc_E2E_distance_Total(self, 
                                latency_core_array_SA_global: np.ndarray, 
                                destination_core_array_SA_global: np.ndarray,
                                latency_core_array_co_global: np.ndarray, 
                                destination_core_array_co_global: np.ndarray,
                                processing_level_list: list,
                                save_flag: int, 
                                save_suffix: str = ""
                                ):
        """
        Compute and optionally save the total end-to-end (E2E) distance from the lowest 
        hierarchy-level nodes up to the top-level core nodes.

        This function iterates over all nodes at the lowest hierarchy level and calls 
        `compute_E2E_path_distance` recursively to determine the cumulative physical distance 
        of all possible paths. The distances are extracted by converting the propagation delays 
        stored in `latency_core_array_...` back to kilometers.

        Args:
        ---------
            latency_core_array_SA_global (np.ndarray): 
                Precomputed hop latencies for SA nodes.
            destination_core_array_SA_global (np.ndarray): 
                Destination mappings for SA nodes.
            latency_core_array_co_global (np.ndarray): 
                Precomputed hop latencies for CO nodes.
            destination_core_array_co_global (np.ndarray): 
                Destination mappings for CO nodes.
            processing_level_list (list): 
                List of network hierarchy levels (e.g., [4, 3, 2, 1]).
            save_flag (int): 
                1 to save the resulting array to disk, 0 otherwise.
            save_suffix (str, optional): 
                Suffix appended to the saved file name. Defaults to "".

        Outputs:
        ---------
            np.ndarray: A 1D array containing the total distances (in km) of all E2E paths found.

        Example:
        ---------
            >>> # Assuming 'self' is an instantiated network object
            >>> total_distances = self.calc_E2E_distance_Total(
            ...     latency_core_array_SA_global=lat_SA, destination_core_array_SA_global=dest_SA,
            ...     latency_core_array_co_global=lat_CO, destination_core_array_co_global=dest_CO,
            ...     processing_level_list=[4, 3, 2, 1], save_flag=1
            ... )
            >>> print(total_distances[:3])
            [250.15 262.04 281.00]
        """
        # HLx hierarchy is traversed top-down; 
        minimum_hierarchy_level = processing_level_list[0]
        minimum_HL_nodes = self.network.hierarchical_levels[f'HL{minimum_hierarchy_level}']['standalone']
        
        E2E_distance_list = []
        for HL_node_IDx in range(len(np.where(np.isin(self.network.all_nodes, minimum_HL_nodes))[0])):
            HL_node_in_net = np.where(np.isin(self.network.all_nodes, minimum_HL_nodes))[0][HL_node_IDx]
            HL_node = minimum_HL_nodes[HL_node_IDx]               
            hierarchy_level_src = self.network.get_standalone_hierarchy_level(HL_node)
            dist_list = self.compute_E2E_path_distance(HL_node_in_net,
                                                        latency_core_array_SA_global,
                                                        destination_core_array_SA_global,
                                                        latency_core_array_co_global,
                                                        destination_core_array_co_global,
                                                        processing_level_list, 
                                                        hierarchy_level_src)
                
            for i in range(len(dist_list)):
                _, latency = dist_list[i]                    
                E2E_distance_list.append(latency)
                
        E2E_distance_list = np.array(E2E_distance_list)

        # Optional: Save to disk in compressed format
        if save_flag:
            np.savez_compressed(
                self.save_directory / f"{self.network.topology_name}_E2E_distance{save_suffix}.npz",
                latency = E2E_distance_list
            )

        return E2E_distance_list

