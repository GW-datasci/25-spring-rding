DC Housing Market Analysis - Environment Setup Guide
This repository contains a data science project analyzing the Washington D.C. housing market using various economic indicators. The analysis includes time series modeling, feature engineering, correlation analysis, and predictive modeling to understand housing price dynamics.
Environment Setup Instructions
This project requires Python 3.11 (not 3.13) due to TensorFlow compatibility constraints. Follow these steps to set up your environment correctly.
1. Install Python 3
First, ensure you have Python 3 installed on your system. You can download it from the official Python website or use a package manager.
Windows:
powershellCopy# Using winget (Windows Package Manager)
winget install Python.Python.3
Mac:
bashCopy# Using Homebrew
brew install python@3
Linux:
bashCopy# Using apt (Ubuntu/Debian)
sudo apt-get install python3
2. Create and Activate a Virtual Environment
Creating a virtual environment helps isolate project dependencies and avoid conflicts with other projects or system packages.
Windows:
powershellCopy# Navigate to your project directory
cd path\to\dc-housing-project

# Create a virtual environment
python3 -m venv venv
python -m venv venv

# Activate the environment
source venv/bin/activate
.\dc_housing_env\Scripts\activate
Mac/Linux:
bashCopy# Navigate to your project directory
cd path/to/dc-housing-project

# Create a virtual environment
python3 -m venv venv

# Activate the environment
source venv/bin/activate
You'll know the environment is activated when you see (dc_housing_env) at the beginning of your command prompt.
3. Install Required Packages
Once your virtual environment is activated, install all required packages using the requirements.txt file:
bashCopy# This must be run first before attempting to run any code
pip install -r requirements.txt
The requirements.txt file includes all necessary packages including:

pandas, numpy (for data manipulation)
matplotlib, seaborn (for visualization)
scikit-learn (for machine learning models)
statsmodels (for statistical analysis)
tensorflow (for deep learning models)
xgboost (for gradient boosting models)

4. Register the Environment with Jupyter
If you're using Jupyter Notebooks for analysis, you need to register your virtual environment as a kernel:
bashCopy# Install ipykernel
pip install ipykernel

# Register the kernel
python -m ipykernel install --user --name=venv --display-name="Python 3(DC Housing)"
5. Running Jupyter Notebooks
Start Jupyter Notebook or Jupyter Lab:
bashCopy# Start Jupyter Notebook
jupyter notebook

# Or start Jupyter Lab if you prefer
jupyter lab
When you open a notebook, make sure to select the "Python 3(DC Housing)" kernel from the kernel menu.
Important Notes

Matplotlib Style: If you encounter style errors with matplotlib, use the updated style syntax:
pythonCopyplt.style.use('seaborn-v0_8-whitegrid')  # For newer matplotlib versions

Data Location: Ensure the data file dc_economic_monthly_dataframe.csv is placed in the project root directory or update the file path in the code accordingly.
Memory Requirements: Some of the machine learning and deep learning models may require significant memory. If you encounter memory issues, try reducing the complexity of models or processing smaller batches of data.

Project Structure

data/: Directory containing the raw and processed datasets
notebooks/: Jupyter notebooks for exploratory data analysis and modeling
src/: Source code modules
requirements.txt: List of required Python packages
README.md: This file

Data Source
This project uses a comprehensive dataset of economic indicators for Washington D.C. from 2000-2024, including:

ZHVI (Zillow Home Value Index)
Unemployment rate
Consumer Price Index (CPI)
Interest rates and mortgage rates
GDP growth
Median household income
Population and demographic indicators

Running the Analysis
After setting up your environment, you can run the analysis by opening the Jupyter notebooks in the notebooks/ directory. Start with the exploratory data analysis (EDA) notebook to understand the dataset and then proceed to the modeling notebooks.
Troubleshooting
Missing Module Errors
If you encounter "module not found" errors, ensure you've activated the virtual environment and installed all requirements:
bashCopy# Activate environment (if not already activated)
.\dc_housing_env\Scripts\activate  # Windows
source dc_housing_env/bin/activate  # Mac/Linux

# Verify pip installation is from the virtual environment
pip -V