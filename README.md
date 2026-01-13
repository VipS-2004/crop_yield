#  Crop Yield Prediction using Rain Water Trapping & Machine Learning

This project is a small experiment where I tried to combine a **DSA concept** with **Machine Learning**.  
I used the well-known **Trapping Rain Water problem** to calculate how much rainwater can be stored based on land elevation, and then used that value to help predict **crop yield**.

---

# What this project is about

- Applying a **DSA algorithm** in a real-world scenario  
- Using **feature engineering** for an ML model  
- Predicting crop yield using simple and explainable logic  

**Tech stack used:**
- Python  
- Pandas  
- Matplotlib  
- Scikit-learn  

---

## How it works

### 1. Rain Water Trapping Logic
I used the **two-pointer approach** to calculate how much water can be trapped between elevation bars.  
This trapped water represents **available irrigation water**, which is useful for agriculture.

The final trapped water value is stored as a new column called **WaterTrapped**.

---

### 2. Data Processing
- Elevation values are converted from strings to integer lists  
- Soil types are encoded into numbers  
- Rainwater trapped is calculated for each row in the dataset  

---

### 3. Machine Learning Model
To keep things simple and interpretable, I used a **Decision Tree Regressor**.

**Input features:**
- WaterTrapped  
- SoilType  
- Rainfall  

**Output:**
- CropYield  

The data is split into training and testing sets to evaluate performance.

---

### 4. Visual Analysis
I added a few basic visualizations using **Matplotlib**:
- Distribution of trapped rainwater  
- Feature importance of the model  
- Comparison of predicted vs actual crop yields  

---




