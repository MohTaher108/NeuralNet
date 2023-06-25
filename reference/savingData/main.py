import numpy as np

def saveToFile(fileName, array, arrayTypeIdentifier):
    np.savetxt(fileName, array, fmt=arrayTypeIdentifier)

def loadFromFile(fileName, arrayType):
    return np.loadtxt(fileName, dtype=arrayType)

a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
saveToFile('myFile.txt', a, '%f')

b = loadFromFile('myFile.txt', float)

print("b = ", b)
print(np.allclose(a, b))