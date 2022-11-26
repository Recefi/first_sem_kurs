

def transpose(array):
    res=[]
    for i in range(len(array[1])):
        current_res=[]
        for j in range(len(array)):
            current_res.append(array[j][i])
        res.append(current_res)
    return res