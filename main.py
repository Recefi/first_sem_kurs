import numpy as np
import matplotlib.pyplot as plt
import source.gen_selection as gs
import source.graphical_interface as gui
import source.input_output as inOut
import source.machine_learning as ml
import source.test_result as tr

# # Либо
# A_jun, B_jun, A_adult, B_adult, stratIndexes = gs.genStrats(200)
# stratData = inOut.writeStrats(A_jun, B_jun, A_adult, B_adult)
# Либо
stratData = inOut.getStratData()
A_jun, B_jun, A_adult, B_adult, stratIndexes = inOut.readStrats(stratData)

# Либо
Fitness, FitIndxs, maxf_ind = gs.calcFitness(A_jun, B_jun, A_adult, B_adult, stratIndexes)
fitData = inOut.writeFitness(Fitness, FitIndxs)
# # Либо
# fitData = inOut.getFitData()
# Fitness, FitIndxs, maxf_ind = inOut.readFitness(fitData)


gui.get_sinss(A_jun[maxf_ind], B_jun[maxf_ind], A_adult[maxf_ind], B_adult[maxf_ind])  # оптимальная синусоида до удаления стратегий

# улучшение корреляции между M1 и M2 удалением стратегий
a, b, xlim, ylim = gui.get_regLine(fitData)
fitData = gui.clean_regLine(fitData, a, b, 10)
gui.get_fixRegLine(fitData, xlim, ylim)

Fitness, FitIndxs, maxf_ind = inOut.readFitness(fitData)  # обновляем фитнес на основе новой fitData
gui.get_sinss(A_jun[maxf_ind], B_jun[maxf_ind], A_adult[maxf_ind], B_adult[maxf_ind])  # оптимальная синусоида после удаления стратегий


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





# корреляция
gui.get_correllation(np.transpose(Fitness)[1:9:1],["M1","M2","M3","M4","M5","M6","M7","M8"])

# # гистограммы нормированных макропараметров
# for i in range(1,9):
#     gui.get_gistogram(np.transpose(selection)[i],"M"+str(i))



# запускаем машинное обучение (внутри функции doSVM все содержимое файла)
lambdas = ml.doSVM(selection)

# запускаем проверку по Тейлору
fitData = inOut.getFitData()

bstPntId = tr.findBestPoint(fitData, lambdas)
print("bstPntId:", bstPntId)

print("Проверка производных:")
tr.calcTeylor(fitData, bstPntId)

print("Сравниваем векторы:")
tr.compareCoefs(bstPntId)
