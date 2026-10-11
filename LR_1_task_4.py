import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from utilities import visualize_classifier

# Вхідний файл, який містить дані
input_file = 'data_multivar_nb.txt'

# Завантаження даних із вхідного файлу
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Створення наївного байєсовського класифікатора
classifier = GaussianNB()

# Тренування класифікатора
classifier.fit(X, y)

# Прогнозування значень для тренувальних даних
y_pred = classifier.predict(X)

# Обчислення якості класифікатора
accuracy = 100.0 * (y == y_pred).sum() / X.shape[0]
print("Accuracy of Naive Bayes classifier =", round(accuracy, 2), "%")

# Візуалізація результатів роботи класифікатора (до перехресної перевірки)
visualize_classifier(classifier, X, y)

# Розбиваємо вибірку на тренувальну (80%) та тестову (20%) частини для незалежної перевірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

# Створюємо новий екземпляр гаусівського наївного байєсовського класифікатора
classifier_new = GaussianNB()

# Навчаємо модель виключно на тренувальній частині даних
classifier_new.fit(X_train, y_train)

# Робимо прогнози для тестового набору даних, які модель ще не бачила
y_test_pred = classifier_new.predict(X_test)

# Обчислюємо відсоток правильних відповідей на тестовій вибірці
accuracy_new = 100.0 * (y_test == y_test_pred).sum() / X_test.shape[0]
print("Accuracy of the new classifier =", round(accuracy_new, 2), "%")

# Візуалізуємо результати роботи класифікатора на тестовому наборі 
visualize_classifier(classifier_new, X_test, y_test)

print("\n--- 3. Оцінка за допомогою перехресної перевірки (Cross-Validation) ---")
# Задаємо кількість блоків (фолдів) для перехресної перевірки
num_folds = 3

# Обчислюємо зважену акуратність (Accuracy) 
accuracy_values = cross_val_score(classifier, X, y, scoring='accuracy', cv=num_folds)
print("Accuracy: " + str(round(100 * accuracy_values.mean(), 2)) + "%")

# Обчислюємо зважену точність (Precision) 
precision_values = cross_val_score(classifier, X, y, scoring='precision_weighted', cv=num_folds)
print("Precision: " + str(round(100 * precision_values.mean(), 2)) + "%")

# Обчислюємо зважену повноту (Recall) 
recall_values = cross_val_score(classifier, X, y, scoring='recall_weighted', cv=num_folds)
print("Recall: " + str(round(100 * recall_values.mean(), 2)) + "%")

# Обчислюємо зважене гармонійне середнє (F1-score) 
f1_values = cross_val_score(classifier, X, y, scoring='f1_weighted', cv=num_folds)
print("F1: " + str(round(100 * f1_values.mean(), 2)) + "%")