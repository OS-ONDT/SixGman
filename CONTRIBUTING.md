# Contributing to SixGman

Thank you for considering contributing to **SixGman**, a Python-based toolkit for optical network planning and simulation!  
We welcome contributions from the community to improve features, documentation, and performance.

---

## 🛠 How to Contribute

There are several ways you can contribute:

1. **Report Bugs**  
   - Check existing [issues](../../issues) to see if the bug is already reported.  
   - Open a new issue with:
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
   - Update the `README.md` or docstrings.

4. **Submit Code Changes**  
   - **Fork the repository**  
   Click the "Fork" button at the top right of this page.  
   - **Clone your fork**

      ```bash
      git clone https://github.com/your-username/SixGman.git
      cd SixGman
      ```
   
     ```bash
     git checkout -b feature/my-feature
     ```
   - **Create a branch**

      ```bash
      git checkout -b my-feature
      ```
   - **Make your changes**
   
      - Follow existing code style and conventions.
      - Write clear, concise commit messages.
   
         ```bash
           git commit -m "Add SNR calculation for multi-band support"
           ```
   - **Push and open a Pull Request**

      ```bash
      git push origin my-feature
      ```

      Go to your fork on GitHub and open a Pull Request.

---

## ✅ Pull Request Guidelines

- Follow **PEP 8** for Python code style.  
- Add **docstrings** to all public functions/classes.  
- Add or update **unit tests** in the `tests/` folder.  
- Ensure that all tests pass:
  
  ```bash
  pytest -v
  ```
  Include a clear description of the problem or feature in your PR.

## 🧪 Running the Project Locally
1. Clone the repo and create a virtual environment:

   ```bash
   git clone https://github.com/UC3M-ONDT/SixGman.git
   cd sixgman
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   .venv\Scripts\activate     # Windows
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run tests to verify everything works:

   ```bash
      pytest -v
   ```

### 📜 Code of Conduct
Please note that this project follows the [Contributor Covenant Code of Conduct](../../CODE_OF_CONDUCT.md).  
Be respectful and collaborative.

