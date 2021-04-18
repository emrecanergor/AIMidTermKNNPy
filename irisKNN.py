import pandas as pnd
import numpy as npy
import math
import operator
 
columnHeaders = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'type']
dataSetIris = pnd.read_csv("iris.xlsx", names = columnHeaders)

#öklid mesafe hesaplama fonksiyonu
def EuclideanDistance(node1, node2, length):
    distance = 0
    for x in range(length):
        distance += npy.square(node1[x] - node2[x])
    return npy.sqrt(distance)

#K-nearest neighbors - en yakın komşuyu hesaplama fonksiyonu
def KNN(trainingSet, testInstance, k):
    distances = {}
    sort = {}
    length = testInstance.shape[1] 
    
    # Her training ve test datası satırları arasında öklid mesafesi ayarlama
    for x in range(len(trainingSet)):
        distance = EuclideanDistance(testInstance, trainingSet.iloc[x], length)
        distances[x] = distance[0]
       
    # Mesafeye göre sort işlemi
    sortedDistances = sorted(distances.items(), key=operator.itemgetter(1)) #Indisleriyle beraber kullanmak için key verilir
    print("Sıralama: ", sortedDistances[:5]) 
   
    neighbors = []
    
    # K tane sıralanmış komşu
    for x in range(k):
        neighbors.append(sortedDistances[x][0])

    counts = {"Iris-setosa" : 0, "Iris-versicolor" : 0, "Iris-virginica" : 0}
    
    # en sık karşılaşılan komşuyu hesaplama
    for x in range(len(neighbors)):
        response = trainingSet.iloc[neighbors[x]][-1]
 
        if response in counts:
            counts[response] += 1
        else:
            counts[response] = 1
   
    sortedVotes = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)
    print(sortedVotes)
    return(sortedVotes[0][0], neighbors)

testSet = [[1.4, 3.6, 3.4, 1.2]]
test = pnd.DataFrame(testSet)
print(dataSetIris["type"].value_counts())

result, neighbor = KNN(dataSetIris, test, 4)
print("Çiçek: ", result)
print("Komşular: ", neighbor)