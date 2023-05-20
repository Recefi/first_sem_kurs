import numpy as np
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

# матрица корреляции
gui.draw_correllation(np.transpose(Fitness)[1:9:1],["M1","M2","M3","M4","M5","M6","M7","M8"])

# оптимальная стратегия
gui.show_sinss(Aj[maxFitId], Bj[maxFitId], Aa[maxFitId], Ba[maxFitId])
# # все синусоиды для которых можно подсчитать формулу фитнеса
# gui.show_all_sinss(stratData.loc[FitIndxs])


# Либо
selection = gs.calcSelection(Fitness)
selData = inOut.collectSelData(selection)
inOut.writeData(selData, "sel_data")
# # Либо
# selData = inOut.readData("sel_data")
# selection = inOut.parseSelData(selData)

# Либо
selection, maxMparamDiffs = gs.normSelection(selection)
normSelData = inOut.collectNormSelData(selection, maxMparamDiffs)
inOut.writeData(normSelData, "norm_sel_data")
# # Либо
# normSelData = inOut.readData("norm_sel_data")
# selection, maxMparamDiffs = inOut.parseNormSelData(normSelData)


# # гистограммы нормированных разностей макропараметров
# for i in range(1,9):
#     gui.show_gistogram(np.transpose(selection)[i],"M"+str(i))


print("запускаем машинное обучение")
norm_machCoefs, intercept = ml.runSVM(selection)  # машинное обучение возвращает "нормированные" лямбда для нормированных макропараметров
machCoefs = norm_machCoefs / maxMparamDiffs  # "разнормируем" машинные коэф-ты

# считаем коэф-ты и косинусы
coefData = tr.getCoefData_2(pqrsData, norm_machCoefs, machCoefs)
inOut.writeData(coefData, "coef_data")
cosines = tr.getCosinesCoef(coefData)
nearPntId = cosines.idxmax()


print("\nвыводим результаты машинного обучения\n")
calcCoefs_mf = coefData.loc[maxFitId]
norm_calcCoefs_mf = calcCoefs_mf * maxMparamDiffs
calcCoefs_n = coefData.loc[nearPntId]
norm_calcCoefs_n = calcCoefs_n * maxMparamDiffs
ml.drawSVM(selection, norm_machCoefs, norm_calcCoefs_mf, norm_calcCoefs_n, intercept, 0, 4)
ml.drawSVM(selection, norm_machCoefs, norm_calcCoefs_mf, norm_calcCoefs_n, intercept, 0, 1)
ml.drawSVM(selection, norm_machCoefs, norm_calcCoefs_mf, norm_calcCoefs_n, intercept, 4, 5)
ml.drawSVM(selection, norm_machCoefs, norm_calcCoefs_mf, norm_calcCoefs_n, intercept, 0, 3)
ml.showAllSVM(selection, norm_machCoefs, norm_calcCoefs_mf, norm_calcCoefs_n, intercept)


print("\nСравнение способов восстановления функции фитнеса:\n")

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

# # оптимальная стратегия по восстановленным коэф-там
# restr_maxFitId = checkCoefData[['restoredFit']].idxmax(axis='index')[0]
# print("\nrestore_maxFitId =", restr_maxFitId)
# gui.show_sinss(Aj[restr_maxFitId], Bj[restr_maxFitId], Aa[restr_maxFitId], Ba[restr_maxFitId])


print("\n")
print("nearPntId =", nearPntId, ", cos:", cosines[nearPntId])
print("maxFitPntId =", maxFitId, ", cos:", cosines[maxFitId])

print("\nСравниваем коэффициенты:")
compareCoefData = tr.compareCoefs(coefData, nearPntId, maxFitId)
inOut.writeData(compareCoefData, "compare_coef_data")
print(compareCoefData)

