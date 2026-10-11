import numpy as np
import matplotlib.pyplot as plt

def visualize_classifier(classifier, X, y):
    # Визначаємо мінімальні та максимальні значення для сітки
    min_x, max_x = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    min_y, max_y = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    
    # Створюємо сітку точок
    step_size = 0.01
    xx, yy = np.meshgrid(np.arange(min_x, max_x, step_size),
                         np.arange(min_y, max_y, step_size))
    
    # Прогнозуємо значення для кожної точки сітки
    output = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    output = output.reshape(xx.shape)
    
    # Будуємо графік
    plt.figure()
    plt.pcolormesh(xx, yy, output, cmap=plt.cm.gray, shading='auto')
    plt.scatter(X[:, 0], X[:, 1], c=y, s=75, edgecolors='k', linewidth=1.5, cmap=plt.cm.Paired)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()