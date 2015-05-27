# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 11:03:25 2015

@author: EmileMathieu
"""

from tools import * 
from collections import Counter

#####################################################################
################ K-plus proche voisin, noyaux #######################
#####################################################################

################### 1.1 Données USPS ################### 

#Permet de charger les données USPS
def load_usps(filename):
    with open(filename,"r") as f:
                 f.readline()
                 data =[ [float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp = np.array(data)
    print tmp.shape
    return tmp[:,1:],tmp[:,0].astype(int)

#Permet de charger une donnée USPS (chiffre manuscrit)
def plot(mat): 
    plt.imshow(mat.reshape((16,16)),interpolation="nearest",cmap=cm.binary)

#Chargement des 2 jeux de données: apprentissage et test
#dataTrain,yTrain = load_usps("2014_tme3_usps_train.txt")
#dataTest, yTest = load_usps("2014_tme3_usps_test.txt")

################### 1.2 K-plus proche voisin ################### 

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
        y=[]        
        for i,x in enumerate(data):
            if i%50==0:
                print i
            arg = np.argsort(((x-self.data)**2).sum(1))
            y.append(arg[0:self.k])
        return y
    def predict(self,data,labeled=True):
        #Renvoie la classe majoritaire 
        #'labeled' permet de différencier le cas où l'on souhaite une moyenne 
        #et celui-ci ou l'on veut un vote majoritaire
        y=np.zeros(data.shape[0]).astype(int)
        res=self.closest(data)
        for i,x in enumerate(res):
            if labeled: 
                y[i]=Counter(self.y[x]).most_common()[0][0]
            else:
                y[i] = sum(self.y[x])/self.k
        return y
        
def Knn_test(k):
    knn = Knn(k)
    knn.fit(dataTrain,yTrain)
    predictedTrain = knn.predict(dataTrain)
    predictedTest = knn.predict(dataTest)
    print "score Train", 1.*(sum(predictedTrain==yTrain))/(1.*len(yTrain))
    print "score Test", 1.*(sum(predictedTest==yTest))/(1.*len(yTest))
    
def Histogram_of_errors(k):
    knn = Knn(k)
    knn.fit(dataTrain,yTrain)
    predictedTest = knn.predict(dataTest)
    hist=np.zeros(10)
    for i,x in enumerate(np.where(predictedTest!=yTest)[0]):
        hist[yTest[x]]=hist[yTest[x]]+1

################### 1.4 Données vélib ###################

# Charge les données velib
import pickle
velib,infoId=pickle.load(file("datavelib.pkl"))

def Lissage(sigma):
    # Lisse par un noyau gaussien les données velib
    interpolation = np.zeros([velib.shape[0],60*24])
    for k in range(interpolation.shape[0]):
        if k%50==0:
            pass
            #print k
        for i in range(interpolation.shape[1]):
            #res = 0.
            #   Z = 0.
            Z = np.exp(-((i-np.arange(0,1440,10))/(sigma))**2)
            res=np.inner(velib[k,:].reshape((7,144)).sum(0),Z)
            interpolation[k,i]=res/Z.sum()
            #for j,y in enumerate(velib[k,:].reshape((7,144)).sum(0)):
            #    Z+=np.exp(-((i-10*j)/(sigma))**2)
            #    res += y*np.exp(-(i-10*j)**2/(sigma)**2)
            #interpolation[k,i]=res/Z`
    return interpolation

def Altitude():
    #Génère un vecteur avec les altitudes des stations velib
    altitude = np.zeros(velib.shape[0])
    for l in range(altitude.shape[0]):
        if infoId[l]['alt']<0: #Une altitude est erronée car négative
            altitude[l] = 30
        else:
            altitude[l] = infoId[l]['alt']
    return altitude

def plot_Lissage(i):
    plt.plot(np.arange(0,10080,10),velib[i,:])
    plt.show()
    plt.plot(np.arange(0,1440,10),velib[i,:].reshape((7,144)).sum(0))
    plt.show()
    plt.plot(np.arange(0,1440,1),interpolation[i,:])
    plt.show()

def Knn_Velib(k):
    knnVelib = Knn(k)
    index = np.arange(interpolation.shape[0])
    np.random.shuffle(index)
    np.random.shuffle(interpolation)
    interpolationTrain=interpolation[]
    interpolationTest=interpolation[]
    knnVelib.fit(interpolationTrain,altitude)
    predictedVelibTrain = knnVelib.predict(interpolationTrain,False)
    predictedVelibTest = knnVelib.predict(interpolationTest,False)
    print "score Train", sum(abs(predictedVelibTrain-altitude)/altitude)/len(altitude)
    print "score Test", sum(abs(predictedVelibTest-altitude)/altitude)/len(altitude)


altitude = Altitude()
interpolation = Lissage(100)
Knn_Velib(2)