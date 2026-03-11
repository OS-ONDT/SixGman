Welcome to SixGman's Documentation!
====================================

**SixGman** is a Python-based toolkit for **optical network modeling, planning, and simulation**.  
It is designed for **researchers, telecom engineers, and students** working in **optical communication networks**.

With SixGman, you can:

- 🏗 **Model optical networks** with nodes, links, and multiple wavelength bands
- 📊 **Analyze optical performance** with configurable physical parameters
- 📡 **Plan traffic routing & capacity allocation** for large-scale networks
- 🎨 **Visualize simulation results** to support network design and optimization



Key Features
-------------

- **Network Modeling** – Define nodes, links, and real/synthetic topologies
- **Optical Band Management** – Handle C-band, L-band, or custom bands
- **Planning & Simulation**  
  - Compute SNR/OSNR and required margins
  - Simulate traffic routing & wavelength assignment (RWA)
  - Evaluate network KPIs for multi-band scenarios
- **Visualization Tools** for network and performance metrics
- **Ready for Research & Teaching** – Easy to extend and integrate into experiments

.. note::

   This documentation is under active development.


Project Structure
------------------

.. code-block:: text

    sixgman/
    ├── src/sixgman/core      # Core classes: Network, Band, PlanningTool
    ├── src/sixgman/utils     # Utility functions and path handling
    ├── tests/                # Unit tests for each module
    ├── examples/             # Jupyter notebooks for simulation examples
    ├── data/                 # Example data files (.mat, .npz)
    ├── results/              # Generated results and network KPIs

----

Contents
---------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   network
   band
   planning
   analysis

.. toctree::
   :maxdepth: 2
   :caption: Contributing

   contributing

.. toctree::
   :maxdepth: 1
   :caption:  Project Info

   License <https://github.com/UC3M-ONDT/SixGman/blob/main/LICENSE>
   GitHub Repository <https://github.com/UC3M-ONDT/SixGman.git>


----

Contact
--------

**Maintainers:** Matin Rafiei Forooshani, Farhad Arpanaei  

📧 Email:  
- `matinrafiei007@gmail.com <mailto:matinrafiei007@gmail.com>`_  
- `farhad.arpanaei@uc3m.es <mailto:farhad.arpanaei@uc3m.es>`_
