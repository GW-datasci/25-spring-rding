
# Capstone Proposal
## Washington, D.C. Housing Price Analysis: Trends, Forecasting, and Affordability
### Proposed by: Raymond Ding
#### Email: rding23@gwu.edu
#### Advisor: Sushovan Majhi
#### The George Washington University, Washington DC  
#### Data Science Program


## 1 Objective:  
 
        The objective of this project is to analyze housing price trends in the Washington, D.C. metro area using 
        Zillow's Home Value Index (ZHVI) data. The goal is to identify key factors influencing housing prices, 
        forecast future trends, and evaluate the affordability of housing in the region. 
        
        This project aims to provide actionable insights for real estate professionals, urban planners, and policymakers.
    

![Figure 1: Example figure](202501_137.png)
*Figure 1: Caption*

## 2 Dataset:  

        The datasets for this project are sourced from Zillow Research's publicly available data platform, 
        which provides reliable and regularly updated metrics on housing markets across the U.S.

        1. Zillow Home Value Index (ZHVI): All Homes, Smoothed, Seasonally Adjusted (Monthly)
            - A measure of typical home values across all housing types.

        2. Zillow Home Value Forecast (ZHVF): All Homes, Smoothed, Seasonally Adjusted, Mid-Tier (Monthly)
            - Predictive insights into future home values.

        3. For-Sale Listings: Median Sale Price
            - The median sale price of homes sold within the Washington, D.C. Metro area.

        4. For-Sale Inventory: All Homes
            - The number of unique for-sale listings active during each month.

        5. Affordability Metrics: New Homeowner Income Needed (20% down)
            - Estimates the annual income required to afford the total monthly payment on a typical home purchase.

        Data is in time-series format and will be preprocessed before modeling.
    

## 3 Rationale:  

        Housing affordability and price trends are critical economic indicators that impact 
        policymakers, investors, and homebuyers. Understanding these trends can guide better 
        decision-making in urban development and housing policy.
    

## 4 Approach:  

        This capstone will follow a structured Data Science approach:

        1. Data Collection and Cleaning
            - Download Zillow datasets and preprocess them (handling missing data, transformations).

        2. Exploratory Data Analysis (EDA)  
            - Visualize historical price trends, neighborhood disparities, and affordability.

        3. Feature Engineering 
            - Create new variables such as price per square foot and housing market indicators.

        4. Model Development
            - Build regression and time-series forecasting models (ARIMA, Prophet, or ML-based).

        5. Visualization and Reporting
            - Create dashboards using Tableau/Power BI, Matplotlib, and Seaborn.

        6. Presentation and Final Report  
            - Summarize findings in a well-documented report with key insights and policy recommendations.
    

## 5 Timeline:  

        - Weeks 1-2: Data Collection and Cleaning  
        - Weeks 3-4: Exploratory Data Analysis and Feature Engineering  
        - Weeks 5-6: Model Development and Evaluation  
        - Week 7: Visualization and Dashboard Creation  
        - Week 8: Final Report, Poster, and Presentation Preparation  
    

## 6 Expected Number Students:  

        This project is designed for just me.
    

## 7 Possible Issues:  

        - Data Completeness: Zillow data may have missing values or inconsistencies.  
          Solution: Apply imputation techniques or drop non-essential missing data.  

        - Model Overfitting: Complex models may not generalize well to unseen data.  
          Solution: Use cross-validation and regularization techniques.  

        - Geospatial Complexity: Mapping trends by neighborhood may be affected by boundary inconsistencies.  
          Solution: Validate geographic data using external sources.  

        - Interpretability: Ensuring that insights are actionable for non-technical stakeholders.  
          Solution: Emphasize clear visualizations and concise reporting.  
    


## Contact
- Author: Sushovan Majhi
- Email: [s.majhi@gwu.edu](Eamil)
- GitHub: [https://github.com/GW-datasci/25-spring-rding](Git Hub rep)
