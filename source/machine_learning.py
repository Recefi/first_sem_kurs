import matplotlib.pyplot as plt
import source.gen_selection as gs
import numpy as np
import source.function as fn
from sklearn import svm
from sklearn import preprocessing
from sklearn.datasets import make_blobs
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import model_selection

def doSVM(selection):
    X = selection[:,1:]
    #print(X)
    Y = selection[:,0]
    #print(Y)

    X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.20)
    # fit the model, don't regularize for illustration purposes
    clf = svm.SVC(kernel="linear")
    clf.fit(X_train, Y_train)
    print(len(X_train))
    #plt.scatter(X[:, 0], X[:, 5], c=Y, s=3, cmap=plt.cm.Paired)

    predictions = clf.predict(X_test)
    print('Точность классификатора:')
    print('     SVM: ', accuracy_score(predictions, Y_test)*100)

    # plot the decision function
    ax1 = plt.gca()
    # DecisionBoundaryDisplay.from_estimator(
    #     clf,
    #     X,
    #     plot_method="contour",
    #     colors="k",
    #     levels=[-1, 0, 1],
    #     alpha=0.5,
    #     linestyles=["--", "-", "--"],
    #     ax=ax,
    # )
    # plot support vectors
    lambdas = clf.coef_[0]
    b = clf.intercept_[0]

    fig, ax = plt.subplots(8,8)
    for i in range(8):
        for j in range(8):
            if i!=j :
                ax[i][j].set(ylim=(-1, 1))
                ax[i][j].scatter(X[:, i], X[:, j], c=Y, s=1, cmap=plt.cm.Paired)
                x_visual = np.linspace(-1, 1)
                y_visual = -(lambdas[i] / lambdas[j]) * x_visual - b / lambdas[j]
                ax[i][j].plot(x_visual, y_visual)

    x_visual = np.linspace(-1,1)
    y_visual = -(lambdas[0] / lambdas[4]) * x_visual - b / lambdas[4]
    ax1.plot(x_visual, y_visual)
    # ax.scatter(
    #     clf.support_vectors_[:, 0],
    #     clf.support_vectors_[:, 5],
    #     s=100,

    #     linewidth=1,
    #     facecolors="none",
    #     edgecolors="k",
    # )

    print(lambdas)

    ax1.scatter(X[:, 0], X[:, 4], c=Y, s=1, cmap=plt.cm.Paired)

    plt.show()

    return lambdas