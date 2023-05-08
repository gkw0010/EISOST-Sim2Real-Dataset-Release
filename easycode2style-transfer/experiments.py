# # # Answer to example
# # import numpy as np
# # from numpy import linalg as la
# # a = np.array([[1, 2, 3, 3], [4, 5, 6, 1], [7, 8, 9, 4]])
# # # b = np.array([[7], [4], [1]])

# # u,sigma,vt = la.svd(a)
# # print('unitary matrix=\n', u)
# # print('diagonal matrix of singular values=\n', sigma)
# # print('another unitary matrix=\n', vt)

# from numpy import *
# import matplotlib.pyplot as plt


# def mean_squared_error(b, m, points):
#     totalError = 0
#     for i in range(0, len(points)):
#         x = points[i, 0]
#         y = points[i, 1]
#         totalError += (y - (m * x + b)) ** 2
#     return totalError

# # Input: current parameters m and b; data points, learning rate;
# # Output: Updated new parameters m and b by using gradient descent.
# def step_gradient_update(b, m, points, learning_rate):
#     N = len(points)
#     gradient_m = 0 
#     gradient_b = 0
#     for i in range(0, N):
#         x = points[i][0]
#         y = points[i][1]
#         gradient_m += -(2/N) * x * (y - ((m*x) + b))
#         gradient_b += -(2/N) * (y - ((m*x) + b))
#     new_b = b - gradient_b * learning_rate
#     new_m = m - gradient_m * learning_rate
#     return new_b, new_m


# # Draw the line
# def draw_plot(points, b, m):
#     x_0 = arange(20, 80)
#     plt.plot(x_0, m * x_0 + b, linestyle="--", color="g", linewidth=1.0)
#     x = points[:, 0]
#     y = points[:, 1]
#     plt.scatter(x, y, s=5, c='r', alpha=1.0, lw=0)
#     plt.savefig("linear_regression.eps", dpi=120)


# def run():
#     points = genfromtxt("data.csv", delimiter=",")
#     learning_rate = 0.0001
#     b = 0 
#     m = 0 
#     num_iterations = 1000
#     print("Starting gradient descent at b = {:.2f}, m = {:.2f}, error = {:.2f}".format(b, m, mean_squared_error(b, m, points)))
#     print("Running...")
#     for i in range(num_iterations):
#         b, m = step_gradient_update(b, m, array(points), learning_rate)
#     print("After {:d} iterations b = {:.2f}, m = {:.2f}, error = {:.2f}".format(num_iterations, b, m, mean_squared_error(b, m, points)))
#     draw_plot(points,b,m)


# if __name__ == '__main__':
#     run()

import numpy as np
from sklearn import svm, datasets
from sklearn.datasets import make_blobs
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
    

# Input: current parameters m and b; data points, learning rate;
# Output: Updated new parameters m and b by using gradient descent.
def train_and_validate(training_data, validation_data, C):
    x, y = training_data
    # train the model
    clf = svm.SVC(kernel="rbf", C=C)
    clf.fit(x, y)
    # validate
    x, y = validation_data
    prediction_y = clf.predict(x)
    metric = accuracy_score(y, prediction_y)
    # [TASK] You should implement metric to evaluate model's performance
    return clf, metric
    


# draw the decision boundary and support vector
def draw_plot(clf, data):
    # plot the decision function
    x, y = data
    plt.scatter(x[:, 0], x[:, 1], c=y, s=30, cmap=plt.cm.Paired)
    
    ax = plt.gca()
    DecisionBoundaryDisplay.from_estimator(
        clf,
        x,
        plot_method="contour",
        colors="k",
        levels=[-1, 0, 1],
        alpha=0.5,
        linestyles=["--", "-", "--"],
        ax=ax,
    )
    # plot support vectors
    ax.scatter(
        clf.support_vectors_[:, 0],
        clf.support_vectors_[:, 1],
        s=100,
        linewidth=1,
        facecolors="none",
        edgecolors="k",
    )
    plt.show()


def run():
    # refer to API doc: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html?highlight=svm
    # set random seet for reproducibility
    np.random.seed(42)
    # generate data
    N = 100
    x, y = make_blobs(n_samples=N, centers=3, random_state=6)
    # add some noise for construct more real data
    x = x * np.random.uniform(0.7, 1.0, size=x.shape)
    # divide training set and validation_set
    sep = int(0.2 * N)
    training_x, training_y = x[:-sep], y[:-sep]
    validation_x, validation_y = x[-sep:], y[-sep:]
    
    print("Load data successfully: total sample number={}".format(N))
    # [TASK] You should try more C choice for better metric on validation dataset
    proposed_C = [1000.0]
    best_metric = 0.
    best_clf = None
    best_C = None
    for C in proposed_C:
        clf, metric = train_and_validate((training_x, training_y), (validation_x, validation_y), C=C)
        print('Training SVM with C={} and metric={:.4f}'.format(C, metric))
        if metric > best_metric:
            best_metric = metric
            best_clf = clf
            best_C = C
    draw_plot(best_clf, (validation_x, validation_y))
    print('The final result: the best C for the validation set is: {}'.format(best_C))


if __name__ == '__main__':
    run()