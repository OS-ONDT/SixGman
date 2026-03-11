Quick Start
============

After installation, you can quickly get started with **network planning and simulation** in SixGman.

----

1️⃣ Launch Example Jupyter Notebooks
-------------------------------------

Navigate to the **examples/** directory and start Jupyter:

.. code-block:: bash

   jupyter notebook examples/

You will find examples such as:

- **MAN157_Singel_HL_Analysis.ipynb** – Basic single-level network planning  
- **MAN157_Full_Hierarchical_50GHz.ipynb** – Multi-level hierarchical network simulation in 50GHz channel spacing

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

----
