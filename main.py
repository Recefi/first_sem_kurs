import numpy as np
import matplotlib.pyplot as plt
import source.gen_selection as gs
import source.graphical_interface as gui
import source.input_output as inOut
import source.machine_learning as ml
import source.test_result as tr

# #     Подсчет и запись cтратегий
# A_jun, B_jun, A_adult, B_adult, stratIndexes = gs.genStrats(200)
# inOut.writeStrats(A_jun, B_jun, A_adult, B_adult)
#       Чтение cтратегий
A_jun, B_jun, A_adult, B_adult, stratIndexes = inOut.readStrats()

# #     Подсчет и запись фитнеса и макропараметров
# Fitness, FitIndxs, maxf_ind = gs.calcFitness(A_jun, B_jun, A_adult, B_adult, stratIndexes)
# inOut.writeFitness(Fitness, FitIndxs)
#       Чтение фитнеса и макропараметров
Fitness, FitIndxs, maxf_ind = inOut.readFitness()

# #     Подсчет и запись итоговой выборки
# selection = gs.calcSelection(Fitness)
# inOut.writeSelection(selection, "sel_data")
#       Чтение итоговой выборки
selection = inOut.readSelection("sel_data")

# #     Подсчет и запись нормированной итоговой выборки
# selection = gs.normSelection(selection)
# inOut.writeSelection(selection, "norm_sel_data")
#       Чтение нормированной итоговой выборки
selection = inOut.readSelection("norm_sel_data")



# оптимальная синусоида
gui.get_sinss(A_jun[maxf_ind], B_jun[maxf_ind], A_adult[maxf_ind], B_adult[maxf_ind])

# улучшение корреляции между M1 и M2
fitData = inOut.getFitData()
a, b, xlim, ylim = gui.get_regLine(fitData)
fitData = gui.clean_regLine(fitData, a, b, 10)
gui.get_fixRegLine(fitData, xlim, ylim)
plt.show()

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
