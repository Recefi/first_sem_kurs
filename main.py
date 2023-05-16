import numpy as np
import matplotlib.pyplot as plt
import source.gen_selection as gs
import source.graphical_interface as gui
import source.input_output as inOut
import source.machine_learning as ml
import source.test_result as tr

# Либо
Aj, Bj, Aa, Ba = gs.genStrats(100)
stratData = inOut.collectStratData(Aj, Bj, Aa, Ba)
inOut.writeData(stratData, "strat_data")
# # Либо
# stratData = inOut.readData("strat_data")
# Aj, Bj, Aa, Ba = inOut.parseStratData(stratData)

# Либо
Fitness, FitIndxs, pqrsData, maxFitId = gs.calcFitness(stratData)
fitData = inOut.collectFitData(Fitness, FitIndxs)
inOut.writeData(fitData, "fit_data")
inOut.writeData(pqrsData, "pqrs_data")
# # Либо
# fitData = inOut.readData("fit_data")
# pqrsData = inOut.readData("pqrs_data")
# Fitness, FitIndxs, maxFitId = inOut.parseFitData(fitData)



# # все синусоиды до удаления стратегий
# gui.get_all_sinss(stratData.loc[FitIndxs])
# # оптимальные синусоиды до удаления стратегий
# gui.get_sinss(Aj[maxFitId], Bj[maxFitId], Aa[maxFitId], Ba[maxFitId])

# # матрица корреляции до удаления стратегий
# gui.get_correllation(np.transpose(Fitness)[1:9:1],["M1","M2","M3","M4","M5","M6","M7","M8"])

# #a, b, xlim78, ylim78 = gui.get_regLine(fitData['M7'], fitData['M8'])

# # исправление корреляции между M1 и M2 удалением стратегий
# fitData = gui.fixCorr(fitData, 'M1', 'M2', 10)
# # исправление корреляции между M5 и M6 удалением стратегий
# fitData = gui.fixCorr(fitData, 'M5', 'M6', 10)

# #gui.get_limRegLine(fitData['M7'], fitData['M8'], xlim78, ylim78)
# #gui.get_limRegLine(fitData['M1'], fitData['M2'], [0, 140], [-140, 0])

# # сохраняем новую fitData в файл
# inOut.writeFitData(fitData, "fit_data_2")
# # обновляем переменные на основе новой fitData
# Fitness, FitIndxs, maxFitId = inOut.parseFitData(fitData)

# матрица корреляции после удаления стратегий
gui.get_correllation(np.transpose(Fitness)[1:9:1],["M1","M2","M3","M4","M5","M6","M7","M8"])

# оптимальные синусоиды после удаления стратегий
gui.get_sinss(Aj[maxFitId], Bj[maxFitId], Aa[maxFitId], Ba[maxFitId])
# # все синусоиды после удаления стратегий
# gui.get_all_sinss(stratData.loc[FitIndxs])

plt.show()


# delIndxs = fitData[fitData['fit'] < 0].index
# fitData = fitData.drop(delIndxs)
# pqrsData = pqrsData.drop(delIndxs)
# inOut.writeData(fitData, "fit_data")
# inOut.writeData(pqrsData, "pqrs_data")

# fitData['fit'] += 50
# inOut.writeData(fitData, "fit_data")


# Либо
selection = gs.calcSelection(Fitness)
inOut.writeSelection(selection, "sel_data")
# # Либо
# selection = inOut.readSelection("sel_data")

# Либо
selection, maxM = gs.normSelection(selection)
inOut.writeSelection(selection, "norm_sel_data")
# # Либо
# selection = inOut.readSelection("norm_sel_data")


# # гистограммы нормированных разностей макропараметров
# for i in range(1,9):
#     gui.get_gistogram(np.transpose(selection)[i],"M"+str(i))


print("запускаем машинное обучение")
machCoef, intercept = ml.runSVM(selection)
#print(machCoef)

coefData = tr.getCoefData_1(pqrsData, machCoef)
inOut.writeData(coefData, "coef_data")

print("выводим результаты машинного обучения")
calcCoef = coefData.loc[maxFitId].to_list()
ml.drawSVM(selection, calcCoef, machCoef, intercept, 0, 4)
ml.showAllSVM(selection, calcCoef, machCoef, intercept)

print("\nПроверка коэффициентов:")
nearPntId = tr.findNearPoint(coefData)
checkCalcCoefData = tr.checkCalcCoef(fitData, pqrsData, maxFitId, nearPntId)
inOut.writeData(checkCalcCoefData, "check_calcCoef_data")
print(checkCalcCoefData)

print("\nСравниваем самые близкие коэффициенты:")
tr.compareCoefs(coefData, nearPntId)
print("\nСравниваем нормированные самые близкие коэффициенты:")
tr.compareNormCoefs(coefData, nearPntId)

