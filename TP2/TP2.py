# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 11:03:25 2015

@author: EmileMathieu
"""

from tools import * 
from collections import Counter

#1 K-plus proche voisin, noyaux 

#1.1 Données USPS

def load_usps(filename):
    with open(filename,"r") as f:
                 f.readline()
                 data =[ [float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp = np.array(data)
    print tmp.shape
    return tmp[:,1:],tmp[:,0].astype(int)

def plot(mat): 
    plt.imshow(mat.reshape((16,16)),interpolation="nearest",cmap=cm.binary)
    
dataTrain,yTrain = load_usps("2014_tme3_usps_train.txt")
dataTest, yTest = load_usps("2014_tme3_usps_test.txt")

# 1.2 K-plus proche voisin
class Knn(Classifier):
    def __init__(self,k=3):
        # k indique le nombre de voisin
        self.k=k
    def fit(self,data,y):
        # Enregistre les donnees
        self.data=data
        self.y=y
    def closest(self,data):
        #Renvoie les indices des k plus proches
        #y=np.zeros([data.shape[0],self.k])
        y=[]        
        for i,x in enumerate(data):
            arg = np.argsort(((x-self.data)**2).sum(1))
            if i%50==0:
                print "i:",i
                #print "self.data.shape", self.data.shape
                #print "data.shape", data.shape
                #print "x.shape",x.shape
            
            y.append(arg[0:self.k])
 
        return y
    def predict(self,data,labeled=True):
        #Renvoie la classe majoritaire 
        y=np.zeros(data.shape[0]).astype(int)
        res=self.closest(data)
        for i,x in enumerate(res):
            #print x
            #print self.y[x]
            if labeled:
                y[i]=Counter(self.y[x]).most_common()[0][0]
            else:
                y[i] = sum(self.y[x])/self.k
            #print y[i]
        return y
        
#knn = Knn(2)
#knn.fit(dataTrain,yTrain)
#predictedTrain = knn.predict(dataTrain)
#predictedTest = knn.predict(dataTest)
#print "score Train", 1.*(sum(predictedTrain==yTrain))/(1.*len(yTrain))
#print "score Test", 1.*(sum(predictedTest==yTest))/(1.*len(yTest))

#####K=2
#Score Train:0.985872994102
#Score Test:0.931738913802
#####K=3
#Score Train:0.986695926485
#Score Test:0.944693572496
#####K=4
#Score Train:0.981209710602
#Score Test:0.943697060289
#####K=5
#Score Train:0.979563845837
#Score Test:0.943198804185
#####K=6
#Score Train:0.974214785352
#Score Test:0.940707523667
#####K=7
#Score Train: 0.974626251543
#Score Test:0.941205779771
#####K=8
#Score Train:0.972157454396
#Score Test:0.93971101146
#####K=9
#Score Train:0.970511589631
#Score Test:0.93971101146


import pickle
velib,infoId=pickle.load(file("datavelib.pkl"))

sigma = 100
#interpolation = np.zeros(1440)
interpolation = np.zeros([velib.shape[0],60*24])

for k in range(interpolation.shape[0]):
    if k%50==0:
        print k
    for i in range(interpolation.shape[1]):
        #res = 0.
        #Z = 0.
        Z = np.exp(-((i-np.arange(0,1440,10))/(sigma))**2)
        res=np.inner(velib[k,:].reshape((7,144)).sum(0),Z)
        interpolation[k,i]=res/Z.sum()
        #for j,y in enumerate(velib[k,:].reshape((7,144)).sum(0)):
        #    Z+=np.exp(-((i-10*j)/(sigma))**2)
        #    res += y*np.exp(-(i-10*j)**2/(sigma)**2)
            
        #interpolation[k,i]=res/Z

altitude = np.zeros(velib.shape[0])
for l in range(altitude.shape[0]):
    if infoId[l]['alt']<0:
        altitude[l] = 30
    else:
        altitude[l] = infoId[l]['alt']

#interpolation = interpolation.T
knnVelib = Knn(5)
print "interpolation.shape", interpolation.shape
knnVelib.fit(interpolation,altitude)
print "knnVelib.data.shape" ,knnVelib.data.shape
print "knnVelib.y.shape", knnVelib.y.shape
predictedVelibTrain = knnVelib.predict(interpolation,False)
print "score Train", sum(abs(predictedVelibTrain-altitude)/altitude)/len(altitude)

plt.plot(np.arange(0,10080,10),velib[1,:])
plt.show()

plt.plot(np.arange(0,1440,10),velib[3,:].reshape((7,144)).sum(0))
plt.show()

plt.plot(np.arange(0,1440,1),interpolation[3,:])
plt.show()
