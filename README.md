# NY Property Fraud Detection Using Unsupervised Machine Learning Models

## Project Overview
This project focuses on identifying potential tax fraud in New York real estate by detecting anomalies in property data using **unsupervised machine learning techniques**. By analyzing property attributes like monetary value, size, and location, the project aims to flag records that deviate significantly from expected norms. 

The workflow includes **data cleaning**, **feature engineering**, **dimensionality reduction**, and the implementation of **anomaly detection algorithms** such as Z-Score Outliers and Autoencoders. Results highlight properties with unusual characteristics, which could indicate tax misrepresentation or fraudulent reporting.

---

## Dataset
The dataset contains **1,070,994 rows and 32 columns** of property data from NYC OpenData, provided by the Department of Finance (DOF). Key attributes include:
- **Numerical Fields:** Monetary values (e.g., FULLVAL, AVLAND), lot and building dimensions.
- **Categorical Fields:** Borough, owner, tax class, and address information.

### Data Cleaning
- **Irrelevant Records Removed:** Excluded government-owned properties and cemeteries.
- **Missing Values Imputed:** Key fields like ZIP, property dimensions, and monetary values were filled using group-specific means or modes.
- **Final Dataset:** 1,044,493 rows of private properties ready for analysis.

---

## Feature Engineering
New variables were created to capture critical property characteristics:
1. **Size Variables:**
   - `lotarea = LTFRONT × LTDEPTH`
   - `bldarea = BLDFRONT × BLDDEPTH`
   - `bldvol = bldarea × STORIES`
2. **Value Ratios:**
   - Ratios of monetary values (e.g., FULLVAL, AVLAND) to size variables.
   - Inverse value ratios for detecting anomalies in properties with unusually low values.
3. **Group-Based Comparisons:**
   - Value variables divided by grouped averages based on ZIP and TAXCLASS.
4. **Size Ratio:** Ratio of `bldarea` to `lotarea` for identifying properties with suspicious dimensions.

A total of **29 new variables** were created for anomaly detection.

---

## Dimensionality Reduction
To handle high-dimensional data, **Principal Component Analysis (PCA)** was applied:
- **Variance Captured:** 75.62% with 5 principal components.
- Data was **z-scaled** before and after PCA to ensure normalization.

---

## Anomaly Detection Algorithms
Two algorithms were used for detecting anomalies:
1. **Z-Score Outliers:**
   - Measures "outlierness" using normalized Minkowski distance.
   - Flags records with large distances from the origin as anomalies.

2. **Autoencoders:**
   - Neural network-based model trained to reconstruct input data.
   - High reconstruction errors indicate anomalous records.

**Final Fraud Score:** Combined the outputs of both methods to rank records by anomaly likelihood.

---

## Results
Top-ranked anomalies were validated through manual inspection:
- **Example 1:** Record with unusually low `AVLAND` value relative to similar properties.
- **Example 2:** Property with an implausible size ratio, where building dimensions exceeded lot size.
- **Example 3:** Record flagged due to abnormal tax-class-specific variables.

These findings highlight properties that warrant further investigation for potential tax fraud.

---

## Recommendations
1. **Client-Specific Adjustments:**
   - Incorporate domain knowledge to refine variable selection and thresholds.
   - Customize anomaly detection criteria based on client needs.
2. **Real-Time Data Integration:**
   - Periodically update the model with new property data.
   - Include market value trends and zoning codes for enhanced accuracy.
3. **Feedback Loop:** 
   - Validate flagged anomalies with clients to improve model precision and reduce false positives.

---

## Files Included
1. **NY DQR.ipynb:** Data quality report, detailing the cleaning process and descriptive statistics.
2. **NY unsupervised fraud.ipynb:** Implementation of anomaly detection models and evaluation.
3. **Project Report 3.pdf:** Full project report, including methodology, results, and recommendations.

---

## Acknowledgments
This project was completed as part of the **DSO 562: Predictive Analytics** course at the University of Southern California. I would like to thank Professor Stephen Coggashell for providing the foundational code and guidance that supported the development of this project. 
