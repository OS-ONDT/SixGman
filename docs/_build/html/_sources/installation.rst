Installation
=============

SixDman can be installed in several ways depending on your workflow:

- **Standard Installation** (from GitHub)  
- **Editable / Development Installation** (recommended for contributors)  

----

Prerequisites
--------------

- Python **3.8+** (recommended 3.9 for Conda)  
- `pip` and optionally `conda`  
- Git installed if installing directly from the repository

----

1️⃣ Standard Installation (Non-editable)
----------------------------------------
You can install the latest version directly from GitHub:

- **Option 1: Using Python venv (Recommended for lightweight setup)**

.. code-block:: bash

   # Create virtual environment
    python -m venv .venv
    
    # Activate it
    source .venv/bin/activate    # Linux/Mac
    .venv\Scripts\activate       # Windows
    
    # Install the package
    pip install git+https://github.com/UC3M-ONDT/SixDman.git

- **Option 2: Using Conda (Recommended for data science users)**

.. code-block:: bash

   # Create a conda environment with Python 3.9+
  conda create -n sixdman-env python=3.9 -y
  
  # Activate environment
  conda activate sixdman-env
  
  # Install the package
  pip install git+https://github.com/UC3M-ONDT/SixDman.git



This will install SixDman and all required dependencies.

----

2️⃣ Development / Editable Installation
----------------------------------------

If you plan to **modify the code** or **contribute**, use editable mode.

**Step 1: Clone the repository**

.. code-block:: bash

  git clone https://github.com/UC3M-ONDT/SixDman.git
  cd sixdman

**Step 2: Create a virtual environment (Recommended)**

- **Option 1: Using Python venv (Recommended for lightweight setup)**

.. code-block:: bash

  # Create virtual environment
  python -m venv .venv
  
  # Activate it
  source .venv/bin/activate    # Linux/Mac
  .venv\Scripts\activate       # Windows
  
  # Install in editable mode
  pip install -e .

- **Option 2: Using Conda (Recommended for data science users)**

.. code-block:: bash

  # Create a conda environment with Python 3.9+
  conda create -n sixdman-env python=3.9 -y
  
  # Activate environment
  conda activate sixdman-env
  
  # Install in editable mode
  pip install -e .

This allows local changes to reflect immediately without reinstallation.

----

Verify Installation
--------------------

To check the installation:

.. code-block:: bash

   python -c "import sixdman; print(sixdman.__version__)"

If no error appears, the package is installed correctly.