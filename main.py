import numpy as np
import matplotlib.pyplot as plt
import source.gen_selection as gs
import source.graphical_interface as gui
import source.input_output as inOut
import source.machine_learning as ml
import source.test_result as tr

# Либо
Aj, Bj, Aa, Ba = gs.genStrats(1000)
stratData = inOut.collectStratData(Aj, Bj, Aa, Ba)
inOut.writeStratData(stratData, "strat_data")
# # Либо
# stratData = inOut.readStratData("strat_data")
# Aj, Bj, Aa, Ba = inOut.parseStratData(stratData)

# Либо
Fitness, FitIndxs, maxf_ind = gs.calcFitness(stratData)
fitData = inOut.collectFitData(Fitness, FitIndxs)
inOut.writeFitData(fitData, "fit_data")
# # Либо
# fitData = inOut.readFitData("fit_data")
# Fitness, FitIndxs, maxf_ind = inOut.parseFitData(fitData)



# # все синусоиды до удаления стратегий
# gui.get_all_sinss(stratData.loc[FitIndxs])
# оптимальные синусоиды до удаления стратегий
gui.get_sinss(Aj[maxf_ind], Bj[maxf_ind], Aa[maxf_ind], Ba[maxf_ind])

# матрица корреляции до удаления стратегий
gui.get_correllation(np.transpose(Fitness)[1:9:1],["M1","M2","M3","M4","M5","M6","M7","M8"])

#a, b, xlim78, ylim78 = gui.get_regLine(fitData['M7'], fitData['M8'])

# исправление корреляции между M1 и M2 удалением стратегий
fitData = gui.fixCorr(fitData, 'M1', 'M2', 10)
# исправление корреляции между M5 и M6 удалением стратегий
fitData = gui.fixCorr(fitData, 'M5', 'M6', 10)

#gui.get_limRegLine(fitData['M7'], fitData['M8'], xlim78, ylim78)
#gui.get_limRegLine(fitData['M1'], fitData['M2'], [0, 140], [-140, 0])

# сохраняем новую fitData в файл
inOut.writeFitData(fitData, "fit_data_2")
# обновляем переменные на основе новой fitData
Fitness, FitIndxs, maxf_ind = inOut.parseFitData(fitData)

# матрица корреляции после удаления стратегий
gui.get_correllation(np.transpose(Fitness)[1:9:1],["M1","M2","M3","M4","M5","M6","M7","M8"])

# оптимальные синусоиды после удаления стратегий
gui.get_sinss(Aj[maxf_ind], Bj[maxf_ind], Aa[maxf_ind], Ba[maxf_ind])
# # все синусоиды после удаления стратегий
# gui.get_all_sinss(stratData.loc[FitIndxs])

plt.show()



# Либо
selection = gs.calcSelection(Fitness)
inOut.writeSelection(selection, "sel_data")
# # Либо
# selection = inOut.readSelection("sel_data")

# Либо
selection = gs.normSelection(selection)
inOut.writeSelection(selection, "norm_sel_data")
# # Либо
# selection = inOut.readSelection("norm_sel_data")


# # гистограммы нормированных макропараметров
# for i in range(1,9):
#     gui.get_gistogram(np.transpose(selection)[i],"M"+str(i))


print("запускаем машинное обучение")
lambdas = ml.doSVM(selection)

print("запускаем проверку по Тейлору")
bstPntId = tr.findBestPoint(fitData, lambdas)
print("bstPntId:", bstPntId)

print("Проверка производных:")
tr.calcTeylor(fitData, bstPntId)

print("Сравниваем векторы:")
tr.compareCoefs(bstPntId)

print("Сравниваем нормированные векторы:")
tr.compareNormCoefs(bstPntId)
