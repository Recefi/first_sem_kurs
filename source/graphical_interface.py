import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import source.input_output as inOut


def show_sinss(Aj, Bj, Aa, Ba):
    fig, ax = plt.subplots()

    xj = np.linspace(0, 1)
    yj = Aj + Bj * np.cos(2 * np.pi * xj)
    ax.plot(xj, yj, c="blue", label="Молодые особи")

    xa = np.linspace(0, 1)
    ya = Aa + Ba * np.cos(2 * np.pi * xa)
    ax.plot(xa, ya, c="red", label="Взрослые особи")

    ax.legend()
    plt.show()

def show_all_sinss(stratData):
    fig, ax = plt.subplots()
    Aj, Bj, Aa, Ba = inOut.parseStratData(stratData)

    for i in Aj.index:
        xj = np.linspace(0, 1)
        yj = Aj[i] + Bj[i] * np.cos(2 * np.pi * xj)
        ax.plot(xj, yj, c="blue")

        xa = np.linspace(0, 1)
        ya = Aa[i] + Ba[i] * np.cos(2 * np.pi * xa)
        ax.plot(xa, ya, c="red")

    plt.show()

def show_comparison_sinss(stratData, maxTrueFitId, maxRestrFitId):
    fig, ax = plt.subplots()
    trueOptStrat = stratData.loc[maxTrueFitId]
    restrOptStrat = stratData.loc[maxRestrFitId]

    xj = np.linspace(0, 1)
    yj = trueOptStrat['Aj'] + trueOptStrat['Bj'] * np.cos(2 * np.pi * xj)
    ax.plot(xj, yj, c="blue", label="Молодые (по исх. функции)")
    xa = np.linspace(0, 1)
    ya = trueOptStrat['Aa'] + trueOptStrat['Ba'] * np.cos(2 * np.pi * xa)
    ax.plot(xa, ya, c="red", label="Взрослые (по исх. функции)")

    yj = restrOptStrat['Aj'] + restrOptStrat['Bj'] * np.cos(2 * np.pi * xj)
    ax.plot(xj, yj, c="green", label="Молодые (по восст. функции)")
    ya = restrOptStrat['Aa'] + restrOptStrat['Ba'] * np.cos(2 * np.pi * xa)
    ax.plot(xa, ya, c="orange", label="Взрослые (по восст. функции)")

    ax.legend()
    plt.show()

def show_gistogram(array, tittle):
    a_min = min(array)
    a_max = max(array)
    fig, histMp = plt.subplots()

    histMp.hist(array, bins=50, linewidth=0.5, edgecolor="white")

    histMp.set(xlim=(-1, 1), xticks=np.linspace(a_min, a_max, 9))
    histMp.set_title(tittle)
    plt.show()

def draw_correllation(array, arg_names):
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
    plt.draw()


def draw_regLine(x, y):
    slope, intercept, r, p, stderr = stats.linregress(x, y)

    fig, ax = plt.subplots()
    ax.scatter(x, y, s=3, label = str(len(x))+" points", color="red")
    ax.plot(x, intercept + slope * x, label = f'corr={r:.2f}', color="blue")
    ax.set_xlabel(x.name)
    ax.set_ylabel(y.name)
    ax.legend()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    plt.draw()
    return intercept, slope, xlim, ylim

def clean_regLine(fitData, xName, yName, a, b, shift):
    x0 = fitData[xName]
    y0 = fitData[yName]
    indexes = []
    for i in x0.index:
        y1 = a - shift + b*x0[i]
        y2 = a + shift + b*x0[i]
        if (y0[i] > y1 and y0[i] < y2):
            indexes.append(i)
    return fitData.drop(indexes)

def draw_limRegLine(x, y, xlim, ylim):
    slope, intercept, r, p, stderr = stats.linregress(x, y)

    fig, ax = plt.subplots()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.scatter(x, y, s=3, label = str(len(x))+" points", color="red")
    ax.plot(x, intercept + slope * x, label = f'corr={r:.2f}', color="blue")
    ax.set_xlabel(x.name)
    ax.set_ylabel(y.name)
    ax.legend()

    plt.draw()

def fixCorr(fitData, xName, yName, shift):
    """
    исправление корреляции между xName и yName 
        удалением стратегий с отступами от линии регрессии на shift
    """
    a, b, xlim, ylim = draw_regLine(fitData[xName], fitData[yName])
    fitData = clean_regLine(fitData, xName, yName, a, b, shift)
    draw_limRegLine(fitData[xName], fitData[yName], xlim, ylim)

    return fitData
