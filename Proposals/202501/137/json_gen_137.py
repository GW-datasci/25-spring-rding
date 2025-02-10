#%%
import json
import os
import shutil

#%%

def save_to_json(data, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8', errors='replace') as output_file:
        json.dump(data, output_file, indent=2)

semester2code = { 
    "sp": "01", "spr": "01", "spring": "01", 
    "su": "02", "sum": "02", "summer": "02", 
    "fa": "03", "fall": "03" 
}
thisfilename = os.path.basename(__file__)  # should match _ver for version, ideally 3-digit string

data_to_save = {
    "Version": "137",
    "Year": "2025",
    "Semester": "Spring",
    "project_name": "Washington, D.C. Housing Price Analysis: Trends, Forecasting, and Affordability",
    
    "Objective": """ 
        The objective of this project is to analyze housing price trends in the Washington, D.C. metro area 
        using macroeconomic, demographic, and housing market indicators. The goal is to identify key 
        factors influencing housing prices, forecast future trends, and evaluate the affordability of 
        housing in the region.
        
        This project aims to provide actionable insights for homebuyers, policymakers, and real estate professionals.
    """,

    "Dataset": """
        The datasets for this project are sourced from various economic and government data repositories. 
        The final dataset is a **time-series master CSV** compiled from multiple sources.

        1. **Housing Market Indicators**  
           - **Zillow Home Value Index (ZHVI)**: Measures median home values.  
           - **Median Sale Price**: Historical sale price trends.  
           - **Housing Inventory**: Tracks supply constraints in D.C.  
           - **Rental Prices (Rent Index)**: Examines rent-buy dynamics.  

        2. **Macroeconomic Indicators**  
           - **Mortgage Rates (30-Year Fixed)**: Impact on affordability.  
           - **Federal Interest Rates**: Influence on lending & investment.  
           - **Consumer Price Index (CPI)**: Measures inflation trends.  
           - **Unemployment Rate**: Tracks job market effects on home demand.  
           - **GDP Growth (National & Local)**: Economic growth proxy.  

        3. **Demographic & Socioeconomic Indicators**  
           - **Median Household Income**: Assesses purchasing power.  
           - **Poverty Rate**: Economic distress signal.  
           - **Population Growth**: Migration & demand shifts.  
           - **Marriage & Divorce Rates**: Correlation with home ownership.  

        Data is in **time-series format (monthly/annual)** and will be preprocessed before modeling.
    """,

    "Rationale": """
        Housing affordability and price trends are critical economic indicators that impact policymakers, 
        investors, and homebuyers. This project aims to quantify the relationships between economic 
        conditions and home prices to provide **data-driven insights** for decision-making.
    """,

    "Approach": """
        This capstone follows a structured Data Science approach:

        1. **Data Collection and Cleaning**  
           - Compile datasets from Zillow, FRED, Census Bureau, and economic databases.  
           - Standardize data to ensure uniformity in time intervals.  

        2. **Exploratory Data Analysis (EDA)**  
           - Visualize historical price trends, affordability metrics, and economic indicators.  
           - Identify correlations between economic shocks and home prices.  

        3. **Feature Engineering**  
           - Transform data to align with predictive modeling.  
           - Create new variables (e.g., mortgage rate-adjusted affordability index).  

        4. **Model Development**  
           - Train regression and time-series forecasting models (ARIMA, Prophet, or ML-based).  

        5. **Visualization and Reporting**  
           - Create interactive dashboards using Tableau/Power BI.  
           - Present findings in a **final report** with key takeaways.  
    """,

    "Timeline": """
        - **Weeks 1-2**: Data Collection and Cleaning  
        - **Weeks 3-4**: Exploratory Data Analysis and Feature Engineering  
        - **Weeks 5-6**: Model Development and Evaluation  
        - **Week 7**: Visualization and Dashboard Creation  
        - **Week 8**: Final Report, Poster, and Presentation Preparation  
    """,

    "Expected Number Students": """
        This project is designed for **one student (me)** but scalable for more.
    """,

    "Possible Issues": """
        - **Data Completeness**: Some datasets may lack historical depth.  
          *Solution*: Use interpolation techniques or external data sources.  

        - **Model Overfitting**: Complex models may not generalize well.  
          *Solution*: Use cross-validation and regularization techniques.  

        - **Temporal Data Alignment**: Merging datasets with different time intervals.  
          *Solution*: Resample data to a **monthly frequency** before modeling.  

        - **Policy & External Shocks**: The market may be affected by unpredictable events (e.g., policy changes, pandemics).  
          *Solution*: Acknowledge limitations and quantify uncertainty.  
    """,

    "Proposed by": "Raymond Ding",
    "Proposed by email": "rding23@gwu.edu",
    "instructor": "Sushovan Majhi",
    "instructor_email": "s.majhi@gwu.edu",
    "github_repo": "https://github.com/GW-datasci/25-spring-rding",
}

os.makedirs(
    os.getcwd() + f'{os.sep}Proposals{os.sep}{data_to_save["Year"]}{semester2code[data_to_save["Semester"].lower()]}{os.sep}{data_to_save["Version"]}',
    exist_ok=True
)
output_file_path = os.getcwd() + f'{os.sep}Proposals{os.sep}{data_to_save["Year"]}{semester2code[data_to_save["Semester"].lower()]}{os.sep}{data_to_save["Version"]}{os.sep}'
save_to_json(data_to_save, output_file_path + "input.json")
shutil.copy(os.path.abspath(__file__), output_file_path)
print(f"✅ Proposal Updated: Data saved to {output_file_path}")