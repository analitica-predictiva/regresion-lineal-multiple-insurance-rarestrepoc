"""
Regresión Lineal Multiple
-----------------------------------------------------------------------------------------

En este laboratorio se entrenara un modelo de regresión lineal multiple que incluye la 
selección de las n variables más relevantes usando una prueba f.

"""
# pylint: disable=invalid-name
# pylint: disable=unsubscriptable-object

import pandas as pd


def pregunta_01():
    """
    Carga de datos.
    -------------------------------------------------------------------------------------
    """
    # Lea el archivo `insurance.csv` y asignelo al DataFrame `df`
    df = pd.read_csv("insurance.csv")

    # Asigne la columna `charges` a la variable `y`.
    y = df["charges"]

    # Asigne una copia del dataframe `df` a la variable `X`.
    X = df.drop(["charges"], axis=1)

    # Retorne `X` y `y`
    return X, y


def pregunta_02():
    """
    Preparación de los conjuntos de datos.
    -------------------------------------------------------------------------------------
    """

    # Importe train_test_split
    from sklearn.model_selection import train_test_split

    # Cargue los datos y asigne los resultados a `X` y `y`.
    X, y = pregunta_01()

    # Divida los datos de entrenamiento y prueba. La semilla del generador de números
    # aleatorios es 12345. Use 300 patrones para la muestra de prueba.
    (X_train, X_test, y_train, y_test,) = train_test_split(
        X,
        y,
        test_size=300,
        random_state=12345,
    )

    # Retorne `X_train`, `X_test`, `y_train` y `y_test`
    return X_train, X_test, y_train, y_test


def pregunta_03():
    """
    Especificación del pipeline y entrenamiento
    -------------------------------------------------------------------------------------
    """

    # Importe make_column_selector
    # Importe make_column_transformer
    # Importe SelectKBest
    # Importe f_regression
    # Importe LinearRegression
    # Importe GridSearchCV
    # Importe Pipeline
    # Importe OneHotEncoder
    from sklearn.compose import make_column_transformer, make_column_selector
    from sklearn.feature_selection import SelectKBest, f_regression
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    pipeline = Pipeline(
        steps=[
            # Paso 1: Transformador de columnas
            (
                "column_transformer",
                make_column_transformer(
                    (
                        OneHotEncoder(),
                        make_column_selector(dtype_include=object),
                    ),
                    remainder="passthrough",
                ),
            ),
            # Paso 2: Selector de características
            (
                "selectKBest",
                SelectKBest(score_func=f_regression),
            ),
            # Paso 3: Modelo de regresión lineal
            (
                "linear_regression",
                LinearRegression(),
            ),
        ],
    )

    # Carga de las variables
    X_train, _, y_train, _ = pregunta_02()

    # Diccionario de parámetros
    param_grid = {
        "selectKBest__k": range(1, 12),
    }

    # Instancia de GridSearchCV
    gridSearchCV = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        refit=True,
        return_train_score=True,
    )

    # Búsqueda de la mejor combinación de regresores
    gridSearchCV.fit(X_train, y_train)

    # Retorno del mejor modelo
    return gridSearchCV


def pregunta_04():
    """
    Evaluación del modelo
    -------------------------------------------------------------------------------------
    """

    # Importe mean_squared_error
    from sklearn.metrics import mean_squared_error

    # Obtenga el pipeline optimo de la pregunta 3.
    gridSearchCV = pregunta_03()

    # Cargue las variables.
    X_train, X_test, y_train, y_test = pregunta_02()

    # Evaluación del modelo
    y_train_pred = gridSearchCV.predict(X_train)
    y_test_pred = gridSearchCV.predict(X_test)

    # Cálculo del error cuadrático medio
    mse_train = mean_squared_error(y_train, y_train_pred).round(2)
    mse_test = mean_squared_error(y_test, y_test_pred).round(2)

    # Retorno de los errores cuadráticos medios
    return mse_train, mse_test
