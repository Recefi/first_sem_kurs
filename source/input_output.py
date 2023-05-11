import pandas as pd
import numpy as np

def readStratData(fileName):
    """Чтение данных стратегий из файла"""
    stratData = pd.read_csv("csv/" + fileName + ".csv", index_col=0)
    return stratData

def parseStratData(stratData):
    """
    Возвращает: A_j, B_j, A_a, B_a 
        в виде pandas series (с сохранением исходных индексов и не только)
    """
    A_j = stratData['Aj']
    B_j = stratData['Bj']
    A_a = stratData['Aa']
    B_a = stratData['Ba']
    return A_j, B_j, A_a, B_a

def collectStratData(A_j, B_j, A_a, B_a):
    """Собираем данные стратегий"""
    stratData = pd.DataFrame({'Aj': A_j, 'Bj': B_j, 'Aa': A_a, 'Ba': B_a})
    return stratData

def writeStratData(stratData, fileName):
    """Запись данных cтратегий в файл"""
    stratData.to_csv("csv/" + fileName + ".csv", index=True)



def readFitData(fileName):
    """Чтение данных фитнеса и макропараметров из файла"""
    fitData = pd.read_csv("csv/" + fileName + ".csv", index_col=0)
    return fitData

def parseFitData(fitData):
    """
    Возвращает: Fitness, FitIndxs, maxf_ind
        FitIndxs[индекс Fitness] = исходный индекс
            maxf_ind в исходных индексах, а не в индексах Fitness
    """
    Fitness = fitData.values
    FitIndxs = fitData.index
    # Получаем индекс макс.значения фитнеса с учетом редактирования fit_data.csv
    maxf_ind = fitData[['fit']].idxmax(axis='index')[0]
    return Fitness, FitIndxs, maxf_ind

def collectFitData(Fitness, FitIndxs):
    """Собираем данные фитнеса и макропараметров"""
    cols = ['fit']
    for i in range(1, 9):
        cols.append('M'+str(i))
    for i in range(1, 9):
        for j in range(i, 9):
            cols.append('M'+str(i) + 'M'+str(j))

    fitData = pd.DataFrame(Fitness, columns=cols, index=FitIndxs)
    return fitData

def writeFitData(fitData, fileName):
    """Запись данных фитнеса и макропараметров в файл"""
    fitData.to_csv("csv/" + fileName + ".csv", index=True)



def writeSelection(selection, fileName):
    """Запись итоговой выборки в файл"""
    cols = ['class']
    for i in range(1, 9):
        cols.append('M'+str(i))
    for i in range(1, 9):
        for j in range(i, 9):
            cols.append('M'+str(i) + 'M'+str(j))

    selData = pd.DataFrame(selection, columns=cols)
    selData.to_csv("csv/" + fileName + ".csv", index=True)

def readSelection(fileName):
    """
    Чтение итоговой выборки из файла
        Существенно экономит время
    """
    selData = pd.read_csv("csv/" + fileName + ".csv")
    selection = selData.values
    return selection

