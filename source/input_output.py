import pandas as pd
import numpy as np

def writeStrats(A_j, B_j, A_a, B_a):
    """Запись cтратегий в файл strat_data"""
    stratData = pd.DataFrame({'Aj': A_j, 'Bj': B_j, 'Aa': A_a, 'Ba': B_a})
    stratData.to_csv("inOut/strat_data.csv", index=True)
    return stratData

def getStratData():
    """Получение данных стратегий из файла strat_data"""
    stratData = pd.read_csv("inOut/strat_data.csv", index_col=0)
    return stratData

def readStrats(stratData):
    """
    Чтение cтратегий
        Возвращает: A_j, B_j, A_a, B_a, Indexes
    """
    A_j = stratData['Aj']
    B_j = stratData['Bj']
    A_a = stratData['Aa']
    B_a = stratData['Ba']
    Indexes = stratData.index
    return A_j, B_j, A_a, B_a, Indexes


def writeFitness(Fitness, FitIndxs):
    """Запись фитнеса и макропараметров в файл fit_data"""
    cols = ['fit']
    for i in range(1, 9):
        cols.append('M'+str(i))
    for i in range(1, 9):
        for j in range(i, 9):
            cols.append('M'+str(i) + 'M'+str(j))

    fitData = pd.DataFrame(Fitness, columns=cols, index=FitIndxs)
    fitData.to_csv("inOut/fit_data.csv", index=True)
    return fitData

def getFitData():
    """Получение данных фитнеса и макропараметров из файла fit_data"""
    fitData = pd.read_csv("inOut/fit_data.csv", index_col=0)
    return fitData

def readFitness(fitData):
    """
    Чтение фитнеса и макропараметров
        Возвращаем: Fitness, Indexes, maxf_ind
    """
    Fitness = fitData.values
    Indexes = fitData.index
    # Получаем индекс макс.значения фитнеса с учетом редактирования fit_data.csv
    maxf_ind = fitData[['fit']].idxmax(axis='index')[0]

    return Fitness, Indexes, maxf_ind


def writeSelection(selection, fileName):
    """Запись итоговой выборки в файл"""
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
    Чтение итоговой выборки из файла
        Существенно экономит время
    """
    selData = pd.read_csv("inOut/" + fileName + ".csv")
    selection = selData.values
    return selection