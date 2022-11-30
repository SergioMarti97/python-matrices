# t-SNE
#
# @autor: Sergio Martí
# @date: 29/11/22
# @see: "https://towardsdatascience.com/an-introduction-to-t-sne-with-python-example-5a3a293108d1"
# @see: "https://www.analyticsvidhya.com/blog/2017/01/t-sne-implementation-r-python/"

# ----------------- #
# --- Librerías --- #
# ----------------- #

import swat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# @see: https://arrow.apache.org/docs/python/getstarted.html
import pyarrow.feather as pa
from time import time
from tabulate import tabulate
# @see: "https://scikit-learn.org/stable/install.html"
from sklearn import (manifold, datasets, decomposition, ensemble, discriminant_analysis, random_projection,
                     feature_selection, preprocessing)


# ----------------- #
# --- Funciones --- #
# ----------------- #

def show_title(title, importance=1):
    """
    Muestra por consola un título
    :param title: el título a mostrar
    :param importance: si es titulo (1), subtitulo (2), sub-subtitulo (3)...
    """
    if importance == 1:
        t = "# --- " + title + " --- #"
        row = "# " + "-"*(len(t) - 4) + " #"
        print(f"{row}\n{t}\n{row}")
    elif importance == 2:
        t = "# --- " + title + " --- #"
        print(f"{t}")
    elif importance == 3:
        t = "# - " + title + " - #"
        print(f"{t}")
    else:
        print(f"# {title} #")


def show_data_frame(df, n_rows=10, n_cols=10):
    """
    Esta función sirve para mostrar parte de un dataframe
    :param df: el dataframe que se quiere mostrar
    :param n_rows: el número de filas a mostrar
    :param n_cols: el número de columnas a mostrar
    """
    print(f"El dataframe tiene {df.shape} filas y columnas")
    if len(df.columns) >= n_cols:
        print(tabulate(df.iloc[:, 0:n_cols].head(n_rows), headers=df.columns[0:n_cols]))
    else:
        print(tabulate(df.head(10), headers=df.columns))


# ------------- #
# --- Datos --- #
# ------------- #

show_title("(1) Adquisición datos")
count_data = pa.read_feather("D:\\Sergio\\Luis Miguel repo\\countData.feather")

if count_data is None:
    print("No se han podido adquirir los datos")
else:
    print("Datos adquiridos satisfactoriamente")
    show_data_frame(count_data)

    # ----------------------- #
    # --- Curado de datos --- #
    # ----------------------- #

    show_title("(2) Pre-procesado")

    # ¿Hay valores nulos?
    show_title("(2.1) Eliminar valores nulos", importance=2)
    print(f"Valores nulos: {count_data.isnull().any().any()}", )

    # Eliminar la variable "gene_id" del dataframe
    col_gene_id = "gene_id"
    show_title(f"(2.2) Extraer la variable \"{col_gene_id}\"", importance=2)
    gene_id = count_data.loc[:, col_gene_id]
    count_data = count_data.drop(col_gene_id, axis=1)

    # Es necesario para aplicar el t-SNE que las carácterísticas sean las columnas y las muestras las filas
    # Para ello, transponemos el dataframe
    show_title("(2.3) Transponer el dataframe", importance=2)
    count_data = count_data.transpose()
    count_data.columns = gene_id
    show_data_frame(count_data)

    # ¿Hay genes que no varían?
    show_title("(2.4) Seleccionar los genes con mayor varianza", importance=2)

    # Guardar en dataframe, las varianzas
    variances = count_data.var()
    variances = pd.DataFrame(variances)
    variances.columns = ["var"]
    show_data_frame(variances)
    pa.write_feather(variances, ".\\variances.feather")

    # Para evitar coger todos los genes, se van a seleccionar los genes con mayor varianza.
    sel = feature_selection.VarianceThreshold()
    count_data_variance = pd.DataFrame(sel.fit_transform(count_data))
    # count_data_variance.columns = count_data.columns
    count_data_variance.index = count_data.index
    show_data_frame(count_data_variance)

    # Escalado de datos
    show_title("(2.5) Escalar los datos", importance=2)
    scaler = preprocessing.StandardScaler()
    scaler.fit(count_data_variance)
    print(f"Número medias para el escalado: {len(scaler.mean_)}")
    count_data_scaled = pd.DataFrame(scaler.transform(count_data_variance))
    # count_data_scaled.columns = count_data.columns
    count_data_scaled.index = count_data.index
    show_data_frame(count_data_scaled)

    # ------------------ #
    # --- Parámetros --- #
    # ------------------ #

    n_samples, n_features = count_data.shape
    n_neighbors = 30
