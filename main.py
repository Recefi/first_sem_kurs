import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import source.gen_selection as gs
import source.graphical_interface as gui
import source.input_output as inOut
import source.machine_learning as ml
import source.test_result as tr

# # Либо
# Aj, Bj, Aa, Ba = gs.genStrats(50)
# stratData = inOut.collectStratData(Aj, Bj, Aa, Ba)
# inOut.writeData(stratData, "strat_data")
# Либо
stratData = inOut.readData("strat_data")
Aj, Bj, Aa, Ba = inOut.parseStratData(stratData)

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
# gui.draw_all_sinss(stratData.loc[FitIndxs])
# # оптимальные синусоиды до удаления стратегий
# gui.draw_sinss(Aj[maxFitId], Bj[maxFitId], Aa[maxFitId], Ba[maxFitId])

# # матрица корреляции до удаления стратегий
# gui.draw_correllation(np.transpose(Fitness)[1:9:1],["M1","M2","M3","M4","M5","M6","M7","M8"])

# #a, b, xlim78, ylim78 = gui.draw_regLine(fitData['M7'], fitData['M8'])

# # исправление корреляции между M1 и M2 удалением стратегий
# fitData = gui.fixCorr(fitData, 'M1', 'M2', 10)
# # исправление корреляции между M5 и M6 удалением стратегий
# fitData = gui.fixCorr(fitData, 'M5', 'M6', 10)

# #gui.draw_limRegLine(fitData['M7'], fitData['M8'], xlim78, ylim78)
# #gui.draw_limRegLine(fitData['M1'], fitData['M2'], [0, 140], [-140, 0])

# # сохраняем новую fitData в файл
# inOut.writeFitData(fitData, "fit_data_2")
# # обновляем переменные на основе новой fitData
# Fitness, FitIndxs, maxFitId = inOut.parseFitData(fitData)

# матрица корреляции после удаления стратегий
gui.draw_correllation(np.transpose(Fitness)[1:9:1],["M1","M2","M3","M4","M5","M6","M7","M8"])

# оптимальные синусоиды после удаления стратегий
gui.draw_sinss(Aj[maxFitId], Bj[maxFitId], Aa[maxFitId], Ba[maxFitId])
# # все синусоиды после удаления стратегий
# gui.draw_all_sinss(stratData.loc[FitIndxs])

plt.show()


# Либо
selection = gs.calcSelection(Fitness)
selData = inOut.collectSelData(selection)
inOut.writeData(selData, "sel_data")
# # Либо
# selData = inOut.readData("sel_data")
# selection = inOut.parseSelData(selData)

# Либо
selection, maxMparams = gs.normSelection(selection)
normSelData = inOut.collectNormSelData(selection, maxMparams)
inOut.writeData(normSelData, "norm_sel_data")
# # Либо
# normSelData = inOut.readData("norm_sel_data")
# selection, maxMparams = inOut.parseNormSelData(normSelData)


# # гистограммы нормированных разностей макропараметров
# for i in range(1,9):
#     gui.show_gistogram(np.transpose(selection)[i],"M"+str(i))


print("запускаем машинное обучение")
norm_machCoefs, intercept = ml.runSVM(selection)
# машинное обучение возвращает "нормированные" лямбда для нормированных макропараметров
# "разнормируем" машинные коэф-ты:
machCoefs = []
for i in range(44):
    machCoefs.append(norm_machCoefs[i]/maxMparams[i])

coefData = tr.getCoefData_2(pqrsData, norm_machCoefs, machCoefs)
inOut.writeData(coefData, "coef_data")
cosines = tr.getCosinesCoef(coefData)
nearPntId = cosines.idxmax()

print("\nвыводим результаты машинного обучения\n")
calcCoefs_mf = coefData.loc[maxFitId]
norm_calcCoefs_mf = calcCoefs_mf * maxMparams
calcCoefs_n = coefData.loc[nearPntId]
norm_calcCoefs_n = calcCoefs_n * maxMparams
ml.drawSVM(selection, norm_machCoefs, norm_calcCoefs_mf, norm_calcCoefs_n, intercept, 0, 4)
ml.drawSVM(selection, norm_machCoefs, norm_calcCoefs_mf, norm_calcCoefs_n, intercept, 0, 1)
ml.drawSVM(selection, norm_machCoefs, norm_calcCoefs_mf, norm_calcCoefs_n, intercept, 4, 5)
ml.drawSVM(selection, norm_machCoefs, norm_calcCoefs_mf, norm_calcCoefs_n, intercept, 0, 3)
ml.drawSVM(selection, norm_machCoefs, norm_calcCoefs_mf, norm_calcCoefs_n, intercept, 4, 7)
ml.showAllSVM(selection, norm_machCoefs, norm_calcCoefs_mf, norm_calcCoefs_n, intercept)

print("\nСравенение способов восстановления функции фитнеса:\n")

checkCoefData = tr.checkCoef(coefData, fitData, pqrsData, maxFitId, nearPntId)
inOut.writeData(checkCoefData, "check_coef_data")
print(checkCoefData)

normCheckCoefData = checkCoefData.copy()
for i in normCheckCoefData.columns:
    normCheckCoefData.loc[:,i] /= np.abs(normCheckCoefData.loc[normCheckCoefData.index[0],i])
inOut.writeData(normCheckCoefData, "norm_check_coef_data")
print(normCheckCoefData)

fitCosines = tr.getFitCosines(checkCoefData).tolist()
fitCosinesData = pd.DataFrame(columns=checkCoefData.columns)
fitCosinesData.loc[0] = fitCosines
fitCosinesData.index = ["cos: "]
print(fitCosinesData)

restr_maxFitId = checkCoefData[['restoredFit']].idxmax(axis='index')[0]
print("\nrestore_maxFitId =", restr_maxFitId)


print("\n")
print("nearPntId =", nearPntId, ", cos:", cosines[nearPntId])
print("maxFitPntId =", maxFitId, ", cos:", cosines[maxFitId])

print("\nСравниваем коэффициенты:")
compareCoefData = tr.compareCoefs(coefData, nearPntId, maxFitId)
inOut.writeData(compareCoefData, "compare_coef_data")
print(compareCoefData)


stratFitData = inOut.collectStratFitData(stratData, checkCoefData)
inOut.writeData(stratFitData, "strat_fit_data")
fitDataByAbsVals = inOut.collectFitDataByAbsVals(checkCoefData)
inOut.writeData(fitDataByAbsVals, "fit_data_byAbsVals")

