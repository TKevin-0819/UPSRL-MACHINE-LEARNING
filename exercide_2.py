# ============================================================
# EXERCISE: Linear and Multiple Linear Regression
# Dataset: FuelConsumptionCo2.csv
# Objective:
#   - Build and analyze a simple linear regression model
#   - Build and analyze a multiple linear regression model
# ============================================================


# ------------------------------------------------------------
# STEP 0: Load required libraries
# ------------------------------------------------------------
# Import the necessary Python libraries for:
#   - Data manipulation (pandas, numpy)
#   - Data visualization (matplotlib)
#   - Machine learning models (scikit-learn)
# Do NOT train any model yet.
import os 
import logging
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score 
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("exercise_02.log", mode='w', encoding="utf-8")
    ]
)
# ------------------------------------------------------------
# STEP 1: Load and inspect the dataset
# ------------------------------------------------------------
# Load the FuelConsumptionCo2.csv file into a pandas DataFrame.
# Display the first few rows to understand the structure.
# Identify:
#   - Feature columns (independent variables)
#   - Target column (dependent variable: CO2EMISSIONS)
# Check data types and verify that there are no missing values.
SOURCE_FILE = os.path.join("inputs", "FuelConsumptionCo2.csv")
OUT_FOLDER = os.path.join(".", "out")

database = pd.read_csv(SOURCE_FILE)

logging.info(f"Archivo procesado: {SOURCE_FILE}")

X = database[database['FUELCONSUMPTION_COMB'] != np.nan]['FUELCONSUMPTION_COMB']
Y = database[database['CO2EMISSIONS'] != np.nan]['CO2EMISSIONS']

logging.info(f"Variable independiente 'FUELCONSUMPTION_COMB':")
logging.info(f"\n{X}\n")

logging.info(f"Variable dependiente 'CO2EMISSIONS':")
logging.info(f"\n{Y}\n")

# ------------------------------------------------------------
# STEP 2: Select relevant variables
# ------------------------------------------------------------
# For teaching purposes, select a reduced subset of variables.
# Typical choices from this dataset are:
#   - ENGINESIZE
#   - CYLINDERS
#   - FUELCONSUMPTION_COMB
#   - CO2EMISSIONS
# Clearly separate:
#   - Input features (X)
#   - Output variable (y)
relevant_features = database[[
    "ENGINESIZE",
    "CYLINDERS",
    "FUELCONSUMPTION_COMB",
    "CO2EMISSIONS"
]]

# ============================================================
# PART 1: SIMPLE LINEAR REGRESSION
# ============================================================

# ------------------------------------------------------------
# STEP 3: Define variables for simple linear regression
# ------------------------------------------------------------
# Choose ONE independent variable that is expected to have a
# strong linear relationship with CO2 emissions.
# Common choice:
#   - FUELCONSUMPTION_COMB
# Reshape the data if required by scikit-learn.
X = relevant_features[relevant_features['FUELCONSUMPTION_COMB'] != np.nan]['FUELCONSUMPTION_COMB']
Y = relevant_features[relevant_features['CO2EMISSIONS'] != np.nan]['CO2EMISSIONS']
# ------------------------------------------------------------
# STEP 4: Split data into training and testing sets
# ------------------------------------------------------------
# Split the dataset into training and testing subsets.
# Use a typical ratio such as:
#   - 80% training
#   - 20% testing
# Set a random_state to ensure reproducibility.
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

logging.info(f"X de entrenamiento generado.")
logging.debug(f"\n{X_train}\n")

logging.info(f"X de pruebas generado.")
logging.debug(f"\n{X_test}\n")

logging.info(f"Y de entrenamiento generado.")
logging.debug(f"\n{Y_train}\n")

logging.info(f"Y de pruebas generado.")
logging.debug(f"\n{Y_test}\n")  

plt.subplot(1,2,1)
plt.scatter(x=X_train, y=Y_train)
plt.title("Datos de entrenamiento")
plt.grid(True)

plt.subplot(1,2,2)
plt.scatter(x=X_test, y=Y_test)
plt.title("Datos de pruebas")
plt.grid(True)

plt.savefig(os.path.join(OUT_FOLDER, "data_split.jpg"))
plt.show()
# ------------------------------------------------------------
# STEP 5: Train the simple linear regression model
# ------------------------------------------------------------
# Create a LinearRegression model.
# Fit the model using the training data.
# At this stage, the model should learn:
#   - Intercept (β0)
#   - Slope (β1)
model = LinearRegression()

X_train = X_train.to_numpy().reshape(-1, 1)
X_test = X_test.to_numpy().reshape(-1, 1)

model.fit(X=X_train, y=Y_train)

logging.info("El modelo linear fue alimentado con 'X_train'.")

# ------------------------------------------------------------
# STEP 6: Analyze the simple linear regression model
# ------------------------------------------------------------
# Extract and display:
#   - Model coefficient
#   - Model intercept
# Interpret the coefficient:
#   - Explain how CO2 emissions change when the selected
#     independent variable increases by one unit.
logging.info(f"Intercept: {model.intercept_}")
logging.info(f"Coefficient: {model.coef_[0]}")

# ------------------------------------------------------------
# STEP 7: Evaluate the simple linear regression model
# ------------------------------------------------------------
# Use the trained model to make predictions on the test set.
# Compute evaluation metrics such as:
#   - Mean Squared Error (MSE)
#   - R² score
# Comment on the quality of the model fit.
y_pred = model.predict(X=X_test)

logging.info(f"MSE: {mean_squared_error(y_true=Y_test, y_pred=y_pred)}")
logging.info(f"R² {r2_score(y_true=Y_test, y_pred=y_pred)}")
# ------------------------------------------------------------
# STEP 8: Visualization
# ------------------------------------------------------------
# Create a scatter plot of the test data.
# Overlay the regression line predicted by the model.
# Label axes clearly and include a title.
plt.scatter(x=X_test, y=Y_test)
plt.plot(X_test, y_pred)
plt.xlabel("Fuel consumption (combined)")
plt.ylabel("CO2 emissions")
plt.title("Prediccion con regresion linear simple ")
plt.grid(True)

plt.savefig(os.path.join(OUT_FOLDER, "srl_model.jpg"))
plt.show()
# ============================================================
# PART 2: MULTIPLE LINEAR REGRESSION
# ============================================================

# ------------------------------------------------------------
# STEP 9: Define variables for multiple linear regression
# ------------------------------------------------------------
# Select multiple independent variables that may influence
# CO2 emissions simultaneously.
# Typical choices include:
#   - ENGINESIZE
#   - CYLINDERS
#   - FUELCONSUMPTION_COMB
# Define X as a matrix of features and y as CO2EMISSIONS.


# ------------------------------------------------------------
# STEP 10: Split data into training and testing sets
# ------------------------------------------------------------
# Perform a new train-test split using the multi-feature dataset.
# Use the same split ratio and random_state as before for
# consistency and fair comparison.


# ------------------------------------------------------------
# STEP 11: Train the multiple linear regression model
# ------------------------------------------------------------
# Create a new LinearRegression model.
# Fit the model using the multiple input features.
# The model will estimate:
#   - One coefficient per feature
#   - A single intercept term


# ------------------------------------------------------------
# STEP 12: Analyze the multiple linear regression model
# ------------------------------------------------------------
# Extract all model coefficients and match each one to its
# corresponding feature.
# Interpret the meaning of each coefficient while assuming
# all other variables remain constant.


# ------------------------------------------------------------
# STEP 13: Evaluate the multiple linear regression model
# ------------------------------------------------------------
# Predict CO2 emissions using the test dataset.
# Calculate evaluation metrics:
#   - Mean Squared Error (MSE)
#   - R² score
# Compare these results with the simple linear regression model.


# ------------------------------------------------------------
# STEP 14: Model comparison and discussion
# ------------------------------------------------------------
# Answer the following questions in comments or markdown:
#   - Does the multiple linear regression improve performance?
#   - Which variable has the strongest influence on CO2 emissions?
#   - Why is multiple linear regression more suitable for this problem?
#   - What assumptions does linear regression make?