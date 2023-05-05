import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import source.gen_selection as gs


def get_sinss(A_j, B_j, A_a, B_a):
    fig, ax = plt.subplots()

    xj = np.linspace(0, 1)
    yj = A_j + B_j * np.cos(2 * np.pi * xj)
    ax.plot(xj, yj, c="blue")

    xa = np.linspace(0, 1)
    ya = A_a + B_a * np.cos(2 * np.pi * xa)
    ax.plot(xa, ya, c="red")

    #plt.ylim([-140, 0])
    plt.show()

def get_gistogram(array, tittle):
    a_min = min(array)
    a_max = max(array)
    fig, histMp = plt.subplots()

    histMp.hist(array, bins=50, linewidth=0.5, edgecolor="white")

    histMp.set(xlim=(-1, 1), xticks=np.linspace(a_min, a_max, 9))
    histMp.set_title(tittle)
    plt.show()

def get_correllation(array, arg_names):
    array_cor=np.round(np.corrcoef(array),2)
    
    fig, cor = plt.subplots()
    im = cor.imshow(array_cor)

    cor.set_xticks(np.arange(8), labels=arg_names)
    cor.set_yticks(np.arange(8), labels=arg_names)

    for i in range(8):
        for j in range(8):
            text = cor.text(j, i, array_cor[i, j],
                        ha="center", va="center", color="r")

    fig.tight_layout()
    plt.show()


def get_regLine(fitData):
    x = fitData['M1']
    y = fitData['M2']
    slope, intercept, r, p, stderr = stats.linregress(x, y)

    fig, ax = plt.subplots()
    ax.scatter(x, y, s=3, label = str(len(x))+" points", color="red")
    ax.plot(x, intercept + slope * x, label = f'corr={r:.2f}', color="blue")
    ax.set_xlabel('M1')
    ax.set_ylabel('M2')
    ax.legend()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    plt.draw()
    return intercept, slope, xlim, ylim

def get_fixRegLine(fitData, xlim, ylim):
    x = fitData['M1']
    y = fitData['M2']
    slope, intercept, r, p, stderr = stats.linregress(x, y)

    fig, ax = plt.subplots()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.scatter(x, y, s=3, label = str(len(x))+" points", color="red")
    ax.plot(x, intercept + slope * x, label = f'corr={r:.2f}', color="blue")
    ax.set_xlabel('M1')
    ax.set_ylabel('M2')
    ax.legend()

    plt.draw()

def clean_regLine(fitData, a, b, eps):
    x0 = fitData['M1']
    y0 = fitData['M2']
    indexes = []
    for i in x0.index:
        y1 = a - eps + b*x0[i]
        y2 = a + eps + b*x0[i]
        if (y0[i] > y1 and y0[i] < y2):
            indexes.append(i)
    return fitData.drop(indexes)

