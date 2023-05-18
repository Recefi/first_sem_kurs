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

Fitness, maxMparams = gs.normArray(Fitness)
normFitData = inOut.collectFitData(Fitness, FitIndxs)
inOut.writeData(normFitData, "norm_fit_data")

# Либо
selection = gs.calcSelection(Fitness)
inOut.writeSelection(selection, "sel_data")
# # Либо
# selection = inOut.readSelection("sel_data")

selection = np.array(selection)

# # Либо
# selection, maxMparams = gs.normArray(selection)
# inOut.writeSelection(selection, "norm_sel_data")
# # Либо
# selection = inOut.readSelection("norm_sel_data")

# # # normMaxMparams = maxMparams / np.abs(maxMparams[0])
# # # maxMparamsData = pd.DataFrame({'maxMparams_normDiff': maxMparams, 'normMaxMparams_normDiff': normMaxMparams})
# # # inOut.writeData(maxMparamsData, "max_Mparams_data_normDiff")

# # # # data111 = inOut.readData("max_Mparams_data_normDiff")
# # # # data222 = inOut.readData("max_Mparams_data_normTrue")
# # # # data333 = pd.concat([data111, data222], axis = 1)
# # # # inOut.writeData(data333, "max_Mparams_data")

# # гистограммы нормированных разностей макропараметров
# for i in range(1,9):
#     gui.get_gistogram(np.transpose(selection)[i],"M"+str(i))


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

print("nearPntId =", nearPntId, ", cos(TaylorCoef^machineCoef):", cosines[nearPntId])
print("maxFitPntId =", maxFitId, ", cos(TaylorCoef^machineCoef):", cosines[maxFitId])

print("\nвыводим результаты машинного обучения\n")
calcCoefs_mf = coefData.loc[maxFitId]
norm_calcCoefs_mf = calcCoefs_mf * maxMparams
calcCoefs_n = coefData.loc[nearPntId]
norm_calcCoefs_n = calcCoefs_n * maxMparams
ml.drawSVM(selection, norm_machCoefs, norm_calcCoefs_mf, norm_calcCoefs_n, intercept, 0, 4)
ml.showAllSVM(selection, norm_machCoefs, norm_calcCoefs_mf, norm_calcCoefs_n, intercept)

print("\nСравенение способов восстановления функции фитнеса:\n")

checkCoefData = tr.checkCoef(coefData, fitData, pqrsData, maxFitId, nearPntId)
inOut.writeData(checkCoefData, "check_coef_data")
print(checkCoefData)

for i in checkCoefData.columns:
    checkCoefData.loc[:,i] /= np.abs(checkCoefData.loc[checkCoefData.index[0],i])
inOut.writeData(checkCoefData, "norm_check_coef_data")
print(checkCoefData)

fitCosines = tr.getFitCosines(checkCoefData).tolist()
fitCosinesData = pd.DataFrame(columns=checkCoefData.columns)
fitCosinesData.loc[0] = fitCosines
fitCosinesData.index = ["cos: "]
print(fitCosinesData)

print("\nСравниваем коэффициенты:")
tr.compareCoefs(coefData, nearPntId, maxFitId)

restr_maxFitId = checkCoefData[['restoredFit']].idxmax(axis='index')[0]
print("restore_maxFitId", restr_maxFitId)
gui.get_sinss(Aj[restr_maxFitId], Bj[restr_maxFitId], Aa[restr_maxFitId], Ba[restr_maxFitId])
plt.show()
