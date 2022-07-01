import pandas as pd
import numpy as np

data = pd.read_csv("table1.csv")
m, n = data.shape
data_std = (data - np.mean (data, axis=0)) / np.std (data, axis=0)
cov_matrix = np.cov (data_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_matrix)
print("Eigenvalues ")
print(eigen_vals)
print()
print("Eigenvectors as columns ")
print(eigen_vecs)
# LHS = np.dot(cov_matrix, eigen_vecs[:, 01)
#RHS = eigen_vals [0] * eigen_vecs [:, 0]
# print(LHS)
# print (RHS)
reduced = np.dot(data_std, eigen_vecs)
print("Reduced")
print(reduced [:, 0])