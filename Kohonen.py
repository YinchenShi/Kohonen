import numpy as np
from random import random
from random import shuffle
from csv import reader
import math

from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt


##read data from csv file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		next(csv_reader) #avoid the first line
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

def Euclidean(p1,p2):
    distance = 0
    for i in range(len(p1)):
        distance += (p1[i] - p2[i])**2
    distance = math.sqrt(distance)
    return distance

def normalization(dataset):
        
	minmax = []
	zip_data = zip(*dataset)  
	for j in zip_data:
		minmax.append({'min': min(j), 'max': max(j)})
		#append min and max data of each column
	for i in range(0, len(dataset)):
		for z in range(len(dataset[i]) - 1):
                        # not include the type
			dataset[i][z] = (dataset[i][z] - minmax[z]['min']) / (minmax[z]['max'] - minmax[z]['min'])
	return dataset

## initialize the the network,
def initialization(n_inputs, n_outputs, saveFile):
    network = [{'weights':[random() for i in range(n_inputs)]} for i in range(n_outputs)]
    #initial weight of ouput layer, using random
    saveFile.write('Initial weights of class1 and class2: ' + str(network) +'\n')
    saveFile.write('Give each feature a random initialweight so that the function can be train without interference.\n')
    return network
    ##return the network we created

##implement competitve algoritm to do competition to gain the weight itself
#this returns the data that doesn't pass 1200
def Winner(network, data):
    dis = 0
    List1 = []
    for i in network:
        dis = 0
        for j in range(len(data)):
            dis += (float(data[j]) - float(i['weights'][j]))**2
        List1.append(dis)
    winner = List1.index(min(List1))
    return winner

###this returns the data that passes 1200
def Competitive(network,data,e1,e2,c1,c2):
    dis = 0
    List1 = []
    List2 = []
    for i in network:
        dis = 0
        for j in range(len(data)):
            dis += (float(data[j]) - float(i['weights'][j]))**2
        List1.append(dis)
    List2 = List1
    winner = List1.index(min(List1))
    if winner == 0:
            c1.append(data)
            e1 += List2[winner]
    else:
            c2.append(data)
            e2 += List2[winner]
    return winner,e1,e2,c1,c2

##Train the data
def Training(network,dataset,l_rate, epoch,saveFile):
    n_epoch = 1
    #training epoch time
    error = 0
    error1 = 0
    #calculate error
    error2 = 0
    class1 = []
    #set two cluster classes
    class2 = []
    while n_epoch != epoch:
        for i in dataset:
            if n_epoch != 999:
                    winner = Winner(network,i)
            else:
                    winner,error1,error2,class1,class2 = Competitive(network,i,error1,error2,class1,class2)
                    error = error1+error2
            for j in range(len(i)):
                network[winner]['weights'][j] += l_rate*(i[j]-network[winner]['weights'][j])
        n_epoch += 1
        l_rate = l_rate-l_rate/n_epoch
        if n_epoch == 1000:
                saveFile.write('Error from Class1: ' + str(error1) + '\n')
                saveFile.write('Error from Class2: ' + str(error2) + '\n')
                saveFile.write('Total Error: ' + str(error) + '\n')
    return network,class1,class2

##kohonen data training algorithm
def Kohonen_Train(dataset, l_rate, epoch,saveFile):
    n_inputs = 3
    n_outputs = 2
    network = initialization(n_inputs, n_outputs, saveFile)
    final_net = Training(network,dataset,l_rate, epoch,saveFile)
    return final_net

##create the file and 3D diagram
def main():
    saveFile = open("Kohonen2.txt", "a")
    ##set our txt file
    filename = 'dataset_noclass.csv'
    dataset = load_csv(filename)
    for i in range(len(dataset)):
        #transfer string to float
        dataset[i] = list(map(eval,dataset[i]))
    shuffle(normalization(dataset))
    #shuffle the data we normalized
    l_rate = 0.5
    epoch = 1000
    final,cluster1,cluster2 = Kohonen_Train(dataset, l_rate, epoch,saveFile)
    saveFile.write('2 Final weight: '+ str(final) + '\n')
    fig = plt.figure()
    #draw the 3d diagram to show the cluster group
    ax = fig.add_subplot(111, projection='3d')
    for x,y,z in cluster1:
        ax.scatter(x, y, z, c='red')
    ##we set the first cluster group represented by red
    for q,w,e in cluster2:
        ax.scatter(q, w, e, c='blue')
    ##we set the first cluster group represented by blue
    for i in final:
        for k,j in i.items():
            ax.scatter(j[0],j[1],j[-1],c='r', marker="*")
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.savefig("3D_Kohonen.png")
    plt.show()
    a = [final[0]['weights'],final[1]['weights']]
    f = open("Kohonen_output.txt",'w')
    count1 =0
    count2 = 0
    for i in range(len(dataset)):
        if Euclidean(dataset[i], a[0]) <= Euclidean(dataset[i], a[1]):
            f.write("Type 1: Datapoint " + str(i+1) + "\n")
            count1 +=1
        else:
            f.write("Type 2: Datapoint " + str(i+1) + "\n")
            count2 +=1
    f.close()
    print(str(count1) +"Type 1, datapoints in total")
    print(str(count2) + "Type 2, datapoints in total")








main()
##call the main fuction
