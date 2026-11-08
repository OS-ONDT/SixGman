Quick Start
============

After installation, you can quickly get started with **network planning and simulation** in SixGman.

----

1️⃣ Launch Example Jupyter Notebooks
-------------------------------------

Navigate to the **examples/** directory and start Jupyter:

.. code-block:: bash

   jupyter notebook examples/

Inside the examples directory, you will find:

- **MAN157** – Simulation scripts for the MAN157 topology.
- **TEFNET24** – Simulation scripts for the TEFNET24 topology.

*(Inside each of these topology folders, the scripts for different CAGRs are separated for both the full hierarchical and HL3 bypassed scenarios.)*

- **journal_plots.ipynb** – Located alongside the MAN157 and TEFNET24 directories, this script contains all the visualization graphs for the JOCN manuscript.

Open a notebook and **run all cells** to see a full demo of:

- Network topology modeling
- Multi-band (C/L/S) traffic planning
- Route and wavelength assignment (RWA)
- KPI visualization

----

2️⃣ Running Unit Tests
-----------------------

Ensure everything works correctly:

.. code-block:: bash

   pytest -v

All tests should pass before starting your experiments.
