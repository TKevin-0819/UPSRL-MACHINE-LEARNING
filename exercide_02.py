# ============================================================
# EXERCISE: Linear and Multiple Linear Regression
# Dataset: FuelConsumptionCo2.csv
# ============================================================

# ------------------------------------------------------------
# STEP 0: Load required libraries
# ------------------------------------------------------------
import os 
import logging
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from mpl_toolkits.mplot3d import Axes3D  # necesario para gráficos 3D

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("exercise_02.log", mode='w', encoding="utf-8")
    ]
)

def plot_regression_3d(model, X, y, feature1="ENGINESIZE", feature2="FUELCONSUMPTION_COMB", filname=None):
    x1 = X[feature1].values
    x2 = X[feature2].values
    y_true = y.values

    x1_range = np.linspace(x1.min(), x1.max(), 30)
    x2_range = np.linspace(x2.min(), x2.max(), 30)
    x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)

    cylinders_mean = X["CYLINDERS"].mean()
    X_grid = np.c_[x1_grid.ravel(), np.full_like(x1_grid.ravel(), cylinders_mean), x2_grid.ravel()]
    y_pred_grid = model.predict(X_grid).reshape(x1_grid.shape)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(x1, x2, y_true, color="blue", alpha=0.5, label="Datos reales")
    ax.plot_surface(x1_grid, x2_grid, y_pred_grid, color="red", alpha=0.4)

    ax.set_xlabel(feature1)
    ax.set_ylabel(feature2)
    ax.set_zlabel("CO2EMISSIONS")
    ax.set_title("Regresión Lineal Múltiple en 3D")

    if filname:
        plt.savefig(filname, dpi=300, bbox_inches="tight")
        print(f"Gráfica guardada en {filname}")
        
    plt.show()

# ------------------------------------------------------------
# STEP 1: Load dataset
# ------------------------------------------------------------
SOURCE_FILE = os.path.join("inputs", "FuelConsumptionCo2.csv")
OUT_FOLDER = os.path.join(".", "out")

os.makedirs(OUT_FOLDER, exist_ok=True)

database = pd.read_csv(SOURCE_FILE)

logging.info(f"Archivo procesado: {SOURCE_FILE}")

X = database["FUELCONSUMPTION_COMB"].dropna()
Y = database["CO2EMISSIONS"].dropna()

logging.info("Variable independiente 'FUELCONSUMPTION_COMB':")
logging.info(f"\n{X}\n")

logging.info("Variable dependiente 'CO2EMISSIONS':")
logging.info(f"\n{Y}\n")

# ------------------------------------------------------------
# STEP 2: Select relevant variables
# ------------------------------------------------------------
relevant_features = database[[
    "ENGINESIZE",
    "CYLINDERS",
    "FUELCONSUMPTION_COMB",
    "CO2EMISSIONS"
]].dropna()

# ============================================================
# PART 1: SIMPLE LINEAR REGRESSION
# ============================================================

X = relevant_features["FUELCONSUMPTION_COMB"]
Y = relevant_features["CO2EMISSIONS"]

# ------------------------------------------------------------
# STEP 4: Split data
# ------------------------------------------------------------
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

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
# STEP 5: Train model
# ------------------------------------------------------------
model = LinearRegression()

X_train = X_train.to_numpy().reshape(-1, 1)
X_test = X_test.to_numpy().reshape(-1, 1)

model.fit(X_train, Y_train)

logging.info("El modelo lineal fue entrenado.")

# ------------------------------------------------------------
# STEP 6: Analyze model
# ------------------------------------------------------------
logging.info(f"Intercept: {model.intercept_}")
logging.info(f"Coefficient: {model.coef_[0]}")

# ------------------------------------------------------------
# STEP 7: Evaluate model
# ------------------------------------------------------------
y_pred = model.predict(X_test)

logging.info(f"MSE: {mean_squared_error(Y_test, y_pred)}")
logging.info(f"R²: {r2_score(Y_test, y_pred)}")

# ------------------------------------------------------------
# STEP 8: Visualization
# ------------------------------------------------------------
plt.scatter(X_test, Y_test)
plt.plot(X_test, y_pred)
plt.xlabel("Fuel consumption (combined)")
plt.ylabel("CO2 emissions")
plt.title("Predicción con regresión lineal simple")
plt.grid(True)

plt.savefig(os.path.join(OUT_FOLDER, "srl_model.jpg"))
plt.show()

# ============================================================
# PART 2: MULTIPLE LINEAR REGRESSION
# ============================================================

X_multiple = relevant_features[[
    "ENGINESIZE",
    "CYLINDERS",
    "FUELCONSUMPTION_COMB",
]]

Y = relevant_features["CO2EMISSIONS"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X_multiple,
    Y,
    test_size=0.2,
    random_state=42
)

model_multiple = LinearRegression()
model_multiple.fit(X_train, Y_train)

logging.info(f"Intercept: {model_multiple.intercept_}")
for feature, coef in zip(X_multiple.columns, model_multiple.coef_):
    logging.info(f"Coefficient of '{feature}': {coef}")

y_pred = model_multiple.predict(X_test)

logging.info(f"MSE: {mean_squared_error(Y_test, y_pred)}")
logging.info(f"R²: {r2_score(Y_test, y_pred)}")

plot_regression_3d(
    model=model_multiple,
    X=X_test,
    y=Y_test,
    feature1="ENGINESIZE",
    feature2="FUELCONSUMPTION_COMB",
    filname=os.path.join(OUT_FOLDER, "mlr_3d_plot.png")
)

corr_features = database[[
    "MODELYEAR",
    "ENGINESIZE",
    "FUELCONSUMPTION_CITY",
    "FUELCONSUMPTION_HWY",
    "FUELCONSUMPTION_COMB",
]].dropna()

correlation = corr_features.corr(method='pearson')
logging.info(f"\n{correlation}")
