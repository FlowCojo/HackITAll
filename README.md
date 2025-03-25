# 🚀 HackITAll

Project created for the **HackITAll** hackathon, built with Python and Flask. Follow the steps below to set up your development environment quickly and cleanly.

## 🔧 Setup

```bash
# Update conda and pip
conda update conda
conda update pip setuptools

# Create a new Conda environment with any Python version you want for example:
conda create -n python311 python==3.11.*
conda activate python311

# Create a local virtual environment (.venv)
python -m venv .venv

# Activate the virtual environment (.venv) on Windows
.venv\Scripts\activate

# Upgrade pip and setuptools inside .venv
python -m pip install --upgrade pip setuptools

# Install project dependencies from requirements.txt
pip install -r requirements.txt

# Run project
flask run

# Swagger localhost
http://127.0.0.1:5000/swagger-ui

