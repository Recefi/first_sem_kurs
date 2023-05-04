import pandas as pd
import numpy as np

def writeStrats(A_j, B_j, A_a, B_a):
    """Заносим A и B в файл strat_data"""
    stratData = pd.DataFrame({'Aj': A_j, 'Bj': B_j, 'Aa': A_a, 'Ba': B_a})
    stratData.to_csv("inOut/strat_data.csv", index=True)
    return stratData.index

def readStrats():
    """
    Считываем A и B из файла strat_data
        Возвращаем: A_j, B_j, A_a, B_a, Indexes
    """
    stratData = pd.read_csv("inOut/strat_data.csv", index_col=0)
    A_j = stratData['Aj']
    B_j = stratData['Bj']
    A_a = stratData['Aa']
    B_a = stratData['Ba']
    Indexes = stratData.index
    return A_j, B_j, A_a, B_a, Indexes

def getStratData():
    """Получаем данные A и B из файла strat_data"""
    stratData = pd.read_csv("inOut/strat_data.csv", index_col=0)
    return stratData


def writeFitness(Fitness, FitIndxs):
    """Заносим Fitness в файл fit_data"""
    cols = ['fit']
    for i in range(1, 9):
        cols.append('M'+str(i))
    for i in range(1, 9):
        for j in range(i, 9):
            cols.append('M'+str(i) + 'M'+str(j))

    fitData = pd.DataFrame(Fitness, columns=cols, index=FitIndxs)
    fitData.to_csv("inOut/fit_data.csv", index=True)

def readFitness():
    """
    Считываем Fitness из файла fit_data
        Возвращаем: Fitness, Indexes, maxf_ind
    """

    fitData = pd.read_csv("inOut/fit_data.csv", index_col=0)
    Fitness = fitData.values
    Indexes = fitData.index

    # Получаем индекс макс.значения фитнеса с учетом редактирования fit_data.csv
    maxf_ind = fitData[['fit']].idxmax(axis='index')[0]

    return Fitness, Indexes, maxf_ind

def getFitData():
    """Получаем данные Fitness из файла fit_data"""
    fitData = pd.read_csv("inOut/fit_data.csv", index_col=0)
    return fitData


def writeSelection(selection, fileName):
    """Заносим selection в файл fileName"""
    cols = ['class']
    for i in range(1, 9):
        cols.append('M'+str(i))
    for i in range(1, 9):
        for j in range(i, 9):
            cols.append('M'+str(i) + 'M'+str(j))

    selData = pd.DataFrame(selection, columns=cols)
    selData.to_csv("inOut/" + fileName + ".csv", index=False)

def readSelection(fileName):
    """
    Считываем selection из файла fileName
        Существенно экономит время
    """
    selData = pd.read_csv("inOut/" + fileName + ".csv")
    selection = selData.values
    return selection