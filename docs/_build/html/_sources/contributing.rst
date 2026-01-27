Contributing to SixDman
=======================

Thank you for considering contributing to **SixDman**, a Python-based toolkit for
optical network planning and simulation!  
We welcome contributions from the community to improve **features**, **documentation**, and **performance**.

.. contents:: Table of Contents
   :local:
   :depth: 2

-----------------------

How to Contribute
-----------------

There are several ways you can contribute to SixDman:

1. **Report Bugs**
    - Check existing `issues <https://github.com/UC3M-ONDT/SixDman/issues>`_ to see if the bug is already reported.
    - Open a new issue and provide:
       - Steps to reproduce
       - Expected behavior
       - Actual behavior
       - Screenshots or error logs (if any)

2. **Suggest New Features**
    - Open a feature request issue describing:
      - Why the feature is needed
      - Possible implementation ideas

3. **Improve Documentation**
    - Fix typos, add examples, or clarify confusing sections.
    - Update the ``README.md`` or docstrings.

4. **Submit Code Changes**  

    - **Fork the repository**  
      Click the "Fork" button at the top right of this page.
    - **Clone your fork**:

      .. code-block:: bash

         git clone https://github.com/your-username/SixDman.git
         cd SixDman

    - **Create a branch**:

      .. code-block:: bash

         git checkout -b my-feature

    - **Make your changes**
      - Follow existing code style and conventions.
      - Write clear, concise commit messages:

        .. code-block:: bash

           git commit -m "Add SNR calculation for multi-band support"

    - **Push and open a Pull Request**:

      .. code-block:: bash

         git push origin my-feature

      Then go to your fork on GitHub and open a **Pull Request**.

-----------------------

Pull Request Guidelines
-----------------------

- Follow **PEP 8** for Python code style.  
- Add **docstrings** to all public functions and classes.  
- Add or update **unit tests** in the ``tests/`` folder.  
- Ensure that all tests pass before submitting your PR:

  .. code-block:: bash

     pytest -v

Include a clear description of the problem or feature in your PR.

-----------------------

Running the Project Locally
---------------------------

1. **Clone the repo and create a virtual environment**:

   .. code-block:: bash

      git clone https://github.com/UC3M-ONDT/SixDman.git
      cd sixdman
      python -m venv .venv
      source .venv/bin/activate  # Linux/Mac
      .venv\Scripts\activate     # Windows

2. **Install dependencies**:

   .. code-block:: bash

      pip install -r requirements.txt

3. **Run tests to verify everything works**:

   .. code-block:: bash

      pytest -v

-----------------------

Code of Conduct
---------------

Please note that this project follows the
`Contributor Covenant Code of Conduct <https://github.com/UC3M-ONDT/SixDman/blob/main/CODE_OF_CONDUCT.md>`_.
Be respectful and collaborative when interacting with the community.