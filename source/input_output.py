import pandas as pd
import numpy as np

def readData(fileName):
    data = pd.read_csv("csv/" + fileName + ".csv", index_col=0)
    return data

def writeData(data, fileName):
    data.to_csv("csv/" + fileName + ".csv", index=True)


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


def parseFitData(fitData):
    """
    Возвращает: Fitness, FitIndxs, maxf_ind
        FitIndxs[индекс Fitness] = исходный индекс
            maxf_ind в исходных индексах, а не в индексах Fitness
    """
    Fitness = fitData.values
    FitIndxs = fitData.index
    # Получаем индекс макс.значения фитнеса с учетом редактирования файла
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


def collectNormFitData(Fitness, FitIndxs, maxMparams):
    """Собираем данные фитнеса и нормированных макропараметров"""
    cols = ['fit']
    for i in range(1, 9):
        cols.append('M'+str(i))
    for i in range(1, 9):
        for j in range(i, 9):
            cols.append('M'+str(i) + 'M'+str(j))

    fitData = pd.DataFrame(Fitness, columns=cols, index=FitIndxs)
    fitData.loc[-1, 'M1':'M8M8'] = maxMparams
    fitData = fitData.sort_index()
    return fitData

def parseNormFitData(fitData):
    """
    Возвращает: Fitness, FitIndxs, maxf_ind, maxMparams
        FitIndxs[индекс Fitness] = исходный индекс
            maxf_ind в исходных индексах, а не в индексах Fitness
                maxMparams в виде массива numpy
    """
    Fitness = fitData.values[1:]
    FitIndxs = fitData.index[1:]
    # Получаем индекс макс.значения фитнеса с учетом редактирования файла
    maxf_ind = fitData[['fit']].idxmax(axis='index')[0]
    maxMparams = fitData.loc[-1, 'M1':'M8M8'].to_numpy()
    return Fitness, FitIndxs, maxf_ind, maxMparams


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
    """Чтение итоговой выборки из файла"""
    selData = pd.read_csv("csv/" + fileName + ".csv")
    selection = selData.values
    return selection


def collectStratFitData(stratData, checkCoefData):
    trueFits = checkCoefData['trueFit']
    restoredFits = checkCoefData['restoredFit']
    stratFitData = pd.concat([stratData, trueFits, restoredFits], axis=1)
    return stratFitData

def collectFitDataByAbsVals(checkCoefData):
    indexes = checkCoefData.index
    def groupByAbsVals(fits):
        i = 0
        k = 0
        fitsByAbsVals = []
        while(i < len(fits)):
            absValFit = []
            for j in range(0, 4):
                if(k % 4 == indexes[i] % 4):
                    absValFit.append(fits[i])
                    i+=1
                    k+=1
                else:
                    absValFit.append(0)
                    k+=1

            best = ""
            if ((absValFit[0] != 0 and absValFit[1] != 0 and absValFit[2] != 0 and absValFit[3] != 0)):
                if ((absValFit[0] < absValFit[3] and absValFit[1] < absValFit[3] and absValFit[2] < absValFit[3])):
                    best = "(-,-)"
                else: 
                    if ((absValFit[0] < absValFit[2] and absValFit[1] < absValFit[2] and absValFit[3] < absValFit[2])):
                        best = "(+,-)"
                    else:
                        if ((absValFit[0] < absValFit[1] and absValFit[2] < absValFit[1] and absValFit[3] < absValFit[1])):
                            best = "(-,+)"
                        else:
                            if ((absValFit[1] < absValFit[0] and absValFit[2] < absValFit[0] and absValFit[3] < absValFit[0])):
                                best = "(+,+)"

            absValFit.append(best)
            fitsByAbsVals.append(absValFit)

        return fitsByAbsVals

    trueFits = checkCoefData['trueFit'].tolist()
    restoredFits = checkCoefData['restoredFit'].tolist()

    trueFitsByAbsVals = groupByAbsVals(trueFits)
    restrFitsByAbsVals = groupByAbsVals(restoredFits)

    trueFitDataByAbsVals = pd.DataFrame(trueFitsByAbsVals, columns=["trueFit(+,+)", "trueFit(-,+)", "trueFit(+,-)", "trueFit(-,-)", "best_trueFit"])
    restrFitDataByAbsVals = pd.DataFrame(restrFitsByAbsVals, columns=["restrFit(+,+)", "restrFit(-,+)", "restrFit(+,-)", "restrFit(-,-)", "best_restrFit"])
    fitDataByAbsVals = pd.concat([trueFitDataByAbsVals, restrFitDataByAbsVals], axis=1)
    return fitDataByAbsVals
