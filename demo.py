import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# --- Step 1: Trap Rain Water Logic ---
class Solution:
    def trap(self, height):
        n = len(height)
        ans = 0
        l, r = 0, n - 1
        lmax, rmax = 0, 0

        while l < r:
            lmax = max(lmax, height[l])
            rmax = max(rmax, height[r])

            if lmax < rmax:
                ans += lmax - height[l]
                l += 1
            else:
                ans += rmax - height[r]
                r -= 1
        return ans

# --- Step 2: Load Dataset ---
df = pd.read_csv("crop_demo_expanded.csv")
 
# Convert elevation string "0 1 0 2 ..." into list of ints
df["Elevations"] = df["Elevations"].apply(lambda x: list(map(int, x.split())))

# Apply trapping rain water algorithm
solver = Solution()
df["WaterTrapped"] = df["Elevations"].apply(lambda h: solver.trap(h))

print("Dataset with Water Trapped Feature:\n", df, "\n")

# --- Step 3: ML Model ---
df["SoilType"] = df["SoilType"].map({"Loamy": 0, "Clay": 1, "Sandy": 2})
X = df[["WaterTrapped", "SoilType", "Rainfall"]]
y = df["CropYield"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Predicted Crop Yields:", y_pred)
print("Actual Crop Yields:", list(y_test))
print("MSE:", mean_squared_error(y_test, y_pred))

# --- Step 4: Visualization with only Matplotlib ---

# 1. Histogram of Water Trapped
plt.figure(figsize=(6,4))
plt.hist(df["WaterTrapped"], bins=10, color="skyblue", edgecolor="black")
plt.title("Distribution of Irrigation Water (Trapped)")
plt.xlabel("Water Trapped")
plt.ylabel("Frequency")
plt.show()

# 2. Feature Importance
plt.figure(figsize=(6,4))
importance = model.feature_importances_
features = X.columns
plt.barh(features, importance, color="lightgreen", edgecolor="black")
plt.title("Feature Importance for Crop Yield Prediction")
plt.xlabel("Importance Score")
plt.show()

# 3. Predicted vs Actual
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, color="orange", edgecolor="black", s=80)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "k--", lw=2)
plt.title("Predicted vs Actual Crop Yields")
plt.xlabel("Actual Crop Yield")
plt.ylabel("Predicted Crop Yield")
plt.show()


