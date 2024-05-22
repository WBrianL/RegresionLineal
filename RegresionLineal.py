class LinearRegression:
    def __init__(self):
        self.coefficients = None

    def fit(self, X, y):
        """
        Entrena el modelo de regresion lineal simple.

        Args:
        X (list): Una lista de valores de la variable independiente.
        y (list): Una lista de valores de la variable dependiente.

        """
        n = len(X)
        sum_x = sum(X)
        sum_y = sum(y)
        sum_x_squared = sum(x ** 2 for x in X)
        sum_xy = sum(X[i] * y[i] for i in range(n))

        # Calcula los coeficientes de la regresion lineal
        denominator = n * sum_x_squared - sum_x ** 2
        a = (sum_y * sum_x_squared - sum_x * sum_xy) / denominator
        b = (n * sum_xy - sum_x * sum_y) / denominator

        self.coefficients = (a, b)

    def predict(self, X):
        """
        Realiza predicciones utilizando el modelo de regresion lineal entrenado.

        Args:
        X (list): Una lista de valores de la variable independiente para los cuales se realizaran predicciones.

        Returns:
        float: El valor predicho para el valor de entrada X.
        """
        if not self.coefficients:
            raise Exception("El modelo no ha sido entrenado.")

        a, b = self.coefficients
        prediction = a + b * X
        return prediction


# Datos de entrenamiento proporcionados
X_train = [651, 762, 856, 1063, 1190, 1298, 1421, 1440, 1518]
y_train = [23, 26, 30, 34, 43, 48, 52, 57, 58]

# Crear y entrenar el modelo de regresion lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Realizar una prediccion para un solo valor
X_test = 1560  # Nuevo valor de entrada para predecir
prediction = model.predict(X_test)
print("Prediction:", prediction)