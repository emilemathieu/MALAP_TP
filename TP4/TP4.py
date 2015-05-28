# -*- coding: utf-8 -*-

from tools import * 
from collections import Counter
from sklearn.cluster import KMeans
#####################################################################
################ Apprentissage non supervisé ########################
#####################################################################
 
#cd /Users/EmileMathieu/Desktop/IMI/MALAP/TP/TP4

def Load_Lena():
    im=plt.imread("lena.png")[:,:,:3] #on garde que les 3 premieres composantes, la tra 
    im_h,im_l,_=im.shape
    pixels=im.reshape((im_h*im_l,3)) #transformation en matrice n*3, n nombre de pixels 
    imnew=pixels.reshape((im_h,im_l,3)) #transformation inverse
    plt.imshow(im) #afficher l’image
    return pixels
    
def KMeans_Lena(k,X):
    KMeans_RGB=KMeans(k)
    KMeans_RGB.fit(X)

Lena_RGB=Load_Lena()
KMeans_RGB=KMeans(32,n_init=5,verbose=1)
KMeans_RGB.fit(Lena_RGB)

Lena_64[:,:]=KMeans_RGB.cluster_centers_[KMeans_RGB.labels_,:]
mnew=Lena_64.reshape((512,512,3))
plt.imshow(mnew)


def RGB_to