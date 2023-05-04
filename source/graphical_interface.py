import matplotlib.pyplot as plt
import numpy as np
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
    plt.draw()

def get_gistogram(array, tittle):
    a_min = min(array)
    a_max = max(array)
    fig, histMp = plt.subplots()

    histMp.hist(array, bins=50, linewidth=0.5, edgecolor="white")

    histMp.set(xlim=(-1, 1), xticks=np.linspace(a_min, a_max, 9))
    histMp.set_title(tittle)
    plt.draw()

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
    plt.draw()