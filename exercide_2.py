import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("exercise_02.log", mode='w', encoding="utf-8")
    ]
)

SOURCE_FILE = os.path.join(".", "inputs", "FuelConsumptionCo2.csv")
OUT_FOLDER = os.path.join(".", "out")
database = pd.read_csv(SOURCE_FILE)

logging.info(f"Archivo procesado: {SOURCE_FILE}")
x =  database[database["FUELCONSUMPTION_COMB"] != np.nan]["FUELCONSUMPTION_COMB"]
y = database[database["FUELCONSUMPTION_COMB"] != np.nan]["CO2EMISSIONS"]

logging.info(f"Variable independiente: FUELCONSUMPTION_COMB")
logging.debug(f"\n{x}\n")

logging.info(f"Variable dependiente: CO2EMISSIONS")
logging.debug(f"\n{y}\n")

relevant_features = database[["ENGINESIZE", "CYLINDERS", "FUELCONSUMPTION_COMB", "CO2EMISSIONS"]]

x =  relevant_features[relevant_features["FUELCONSUMPTION_COMB"] != np.nan]["FUELCONSUMPTION_COMB"]
y = relevant_features[relevant_features["FUELCONSUMPTION_COMB"] != np.nan]["CO2EMISSIONS"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

logging.info("X de entrenamiento generado.")
logging.debug(f"\n{x_train}\n")

logging.info("X de prueba generado.")
logging.debug(f"\n{x_test}\n")

logging.info("y de entrenamiento generado.")
logging.debug(f"\n{y_train}\n")

logging.info("y de prueba generado.")
logging.debug(f"\n{y_test}\n")

# ------------------------------------------------------------
# Visualización de datos de entrenamiento y prueba
# ------------------------------------------------------------
plt.figure(figsize=(8, 6))

# Datos de entrenamiento
plt.subplot(2, 1, 1)
plt.scatter(x_train, y_train, color="blue")
plt.title("Datos de entrenamiento")
plt.xlabel("Fuel Consumption (COMB)")
plt.ylabel("CO2 Emissions")
plt.grid(True)

# Datos de prueba
plt.subplot(2, 1, 2)
plt.scatter(x_test, y_test, color="green")
plt.title("Datos de prueba")
plt.xlabel("Fuel Consumption (COMB)")
plt.ylabel("CO2 Emissions")
plt.grid(True)

plt.tight_layout()
plt.savefig("out/train_test_scatter.png")   # guarda la imagen en archivo
logging.info("Gráfica guardada en out/train_test_scatter.png")


model = LinearRegression()

x_train = x_train.to_numpy().reshape(-1, 1)
x_test = x_test.to_numpy().reshape(-1, 1)
model.fit(X=x_train, y=y_train)

logging.info("El modelo linear fue alimentado ")
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
logging.info(f"Coeficient: {model.coef_[0]}")

# ------------------------------------------------------------
# STEP 7: Evaluate the simple linear regression model
# ------------------------------------------------------------
# Use the trained model to make predictions on the test set.
# Compute evaluation metrics such as:
#   - Mean Squared Error (MSE)
#   - R² score
# Comment on the quality of the model fit.

y_pred = model.predict(X=x_test)

logging.info(f"MSE: {mean_squared_error(y_true=y_test, y_pred=y_pred)}")
logging.info(f"R2: {r2_score(y_true=y_test, y_pred=y_pred)}")


# ------------------------------------------------------------
# STEP 8: Visualization
# ------------------------------------------------------------
# Create a scatter plot of the test data.
# Overlay the regression line predicted by the model.
# Label axes clearly and include a title.


plt.scatter(x_train, y_train, color="blue")
plt.plot(x_test, y_pred)
plt.title("Predicción con regresión linear limpio")
plt.xlabel("Fuel Consumption (COMB)")
plt.ylabel("CO2 Emissions")
plt.grid(True)

plt.tight_layout()
plt.savefig("out/regression_prediction.png")   # crea carpeta outputs si quieres organizar
logging.info("Gráfica guardada en out/regression_prediction.png")