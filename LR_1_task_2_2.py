import numpy as np
from sklearn import preprocessing

input_data = np.array(
    [[2.5, -1.6, -6.1],
     [-2.4, -1.2, 4.3], 
     [3.2, 3.1, 6.1], 
     [-4.4, 1.4, -1.2]]
)

# Бінаризація
data_binarized = preprocessing.Binarizer(threshold=2.5).transform(input_data)
print("\nBinarized data:\n", data_binarized)

# Виведення середнього значення та стандартного відхилення ДО обробки
print("\nBEFORE: ")
print("Mean =", input_data.mean(axis=0))
print("Std deviation =", input_data.std(axis=0))

# Виключення середнього 
data_scaled = preprocessing.scale(input_data)
print("\nAFTER: ")
print("Mean =", data_scaled.mean(axis=0))
print("Std deviation =", data_scaled.std(axis=0))

# Масштабування
data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled_minmax = data_scaler_minmax.fit_transform(input_data)
print("\nMin max scaled data:\n", data_scaled_minmax)

# Нормалізація
data_normalized_l1 = preprocessing.normalize(input_data, norm="l1")
data_normalized_l2 = preprocessing.normalize(input_data, norm="l2")

print("\nL1 normalized data:\n", data_normalized_l1)
print("\nL2 normalized data:\n", data_normalized_l2)