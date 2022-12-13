import matplotlib.pyplot as plt
import source.gen_selection as gs
import numpy as np
import source.function as fn
from sklearn import svm
from sklearn import preprocessing
from sklearn.datasets import make_blobs
from sklearn.inspection import DecisionBoundaryDisplay





X=gs.selection[:,1:]
print(X)
Y = np.transpose(gs.selection[:,0])
print(Y)

# we create 40 separable points
# X = []
# y = []
# for v in range(len(gs.Macroparameters[1])):
#     for w in range(len(gs.Macroparameters[1])):
#         x=[]
#         if(w==v):
#             continue
#         if (gs.classification_Table[v][w]!=0):
#             for i in range(8):
#                 x.append(gs.Macroparameters[i][v]-gs.Macroparameters[i][w])
#             for i in range(8):
#                 for j in range(i,8):
#                     x.append(gs.M_sqr[i][j][v]-gs.M_sqr[i][j][w])
#             y.append(gs.classification_Table[v][w])
#             X.append(x)
                    


                    
# for i in range(len(X)):
#     print(len(X[i]))

# print(X[1])
# X= fn.transpose(X)
# print(len(X[1]))
# print(X[1])
# res=[[]for i in range(len(X))]
# for i in range(len(X)):
#     max_param = max([abs(mp) for mp in X[i]])
#     for j in range(len(X[i])):
#         res[i].append(X[i][j]/max_param)
# X =res
# print(X[1])
# X = fn.transpose(res)





# fit the model, don't regularize for illustration purposes
clf = svm.SVC(kernel="linear", C=1000)
clf.fit(X, Y)
print(len(X))
#plt.scatter(X[:, 0], X[:, 5], c=Y, s=3, cmap=plt.cm.Paired)

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