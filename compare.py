import numpy as np

def loadFromFile(fileName, arrayType):
    return np.loadtxt(fileName, dtype=arrayType)


bias_changes_list_3 = loadFromFile('files/UGD_bias_changes_list_3.txt', float)
bias_changes_list_2 = loadFromFile('files/UGD_bias_changes_list_2.txt', float)
bias_changes_list_1 = loadFromFile('files/UGD_bias_changes_list_1.txt', float)

weight_changes_list_3 = loadFromFile('files/UGD_weight_changes_list_3.txt', float)
weight_changes_list_2 = loadFromFile('files/UGD_weight_changes_list_2.txt', float)
weight_changes_list_1 = loadFromFile('files/UGD_weight_changes_list_1.txt', float)

bias_changes_3 = loadFromFile('files/bias_changes_3.txt', float)
bias_changes_2 = loadFromFile('files/bias_changes_2.txt', float)
bias_changes_1 = loadFromFile('files/bias_changes_1.txt', float)

weight_changes_3 = loadFromFile('files/weight_changes_3.txt', float)
weight_changes_2 = loadFromFile('files/weight_changes_2.txt', float)
weight_changes_1 = loadFromFile('files/weight_changes_1.txt', float)

print("bias_3 = ", np.allclose(bias_changes_list_3, bias_changes_3))
print("bias_2 = ", np.allclose(bias_changes_list_2, bias_changes_2))
print("bias_1 = ", np.allclose(bias_changes_list_1, bias_changes_1))

print("weight_3 = ", np.allclose(weight_changes_list_3, weight_changes_3))
print("weight_2 = ", np.allclose(weight_changes_list_2, weight_changes_2))
print("weight_1 = ", np.allclose(weight_changes_list_1, weight_changes_1))