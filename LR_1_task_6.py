import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score
from utilities import visualize_classifier

# Завантаження набору даних 
input_file = 'data_multivar_nb.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Розбивка даних на навчальний та тестовий набори (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

# Створення та тренування класифікатора SVM (лінійне ядро)
classifier = svm.SVC(kernel='linear', C=1.0)
classifier.fit(X_train, y_train)

# Прогнозування для тестових даних
y_test_pred = classifier.predict(X_test)

# Обчислення якості для тестового набору
accuracy = 100.0 * (y_test == y_test_pred).sum() / X_test.shape[0]
print("Accuracy of SVM classifier =", round(accuracy, 2), "%")

# Візуалізація результатів роботи SVM на тестових даних
visualize_classifier(classifier, X_test, y_test)

# Обчислення показників якості за допомогою перехресної перевірки 
num_folds = 3

# Акуратність
accuracy_values = cross_val_score(classifier, X, y, scoring='accuracy', cv=num_folds)
print("Accuracy: " + str(round(100 * accuracy_values.mean(), 2)) + "%")

# Точність
precision_values = cross_val_score(classifier, X, y, scoring='precision_weighted', cv=num_folds)
print("Precision: " + str(round(100 * precision_values.mean(), 2)) + "%")

# Повнота
recall_values = cross_val_score(classifier, X, y, scoring='recall_weighted', cv=num_folds)
print("Recall: " + str(round(100 * recall_values.mean(), 2)) + "%")

f1_values = cross_val_score(classifier, X, y, scoring='f1_weighted', cv=num_folds)
print("F1: " + str(round(100 * f1_values.mean(), 2)) + "%")