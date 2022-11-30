# Librerías
import numpy as np
import pandas as pd
import pyarrow.feather as pa
from tabulate import tabulate

# Leer los datos
mat_cor = pa.read_feather("D:\\Sergio\\Luis Miguel repo\\feature selection\\mCorCountDataHighVarLight.feather")
gene_names = pa.read_feather("D:\\Sergio\\Luis Miguel repo\\feature selection\\lHighVarGenes.feather")

# Transfromar la matriz a una matriz de numpy
mat_cor = mat_cor.to_numpy()

# Indices
s_col = 0
e_col = 10
s_row = 0
e_row = 10

# threshold
threshold = 0.5

# Partir la matriz en una submatriz
m = mat_cor[s_row:e_row, s_col:e_col]

# Si es una submatriz, que cae en la diagonal, se trianguliza
# m = np.triu(m)

# Mostrar la matriz
print(tabulate(m))

biglist = list()

for i_row in range(e_row - s_row + 1):
    for i_col in range(e_col - s_col + 1):
        real_col = (i_col + s_col)
        real_row = (i_row + s_row)
        corr = mat_cor[real_row, real_col]
        if real_row != real_col and \
                corr != 0 and \
                corr > threshold:
            entry = list()
            entry.append(gene_names.at[real_row, 1].val)
            entry.append(gene_names.at[real_col, 1].val)
            entry.append(mat_cor[real_row, real_col])

            biglist.append(entry)


df = pd.DataFrame(biglist, columns=['Gene 1', 'Gene 2', "corr"])

df
