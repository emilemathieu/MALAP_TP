# -*- coding: utf-8 -*-

# # TP : Arbres de décision et fôrets aléatoires
# 
# 
# ## Résumé
# 
# Un arbre de décision est un modèle de classification hiérarchique : à chaque noeud de l'arbre
# est associé un test sur une des dimensions $x_i$ de la forme $x_i \{\leq,~ >,~ = \} s$ ($s$ une valeur réelle) qui indique le noeud fils qui doit être sélectionné (par exemple pour un arbre binaire, le fils gauche quand le test est vrai, le fils droit sinon). A chaque feuille de l'arbre est associée une étiquette. Ainsi, la classification d'un exemple consiste en une succession de tests sur les valeurs des dimensions de l'exemple, selon un chemin dans l'arbre de la racine à une des feuilles. La feuille atteinte donne la classe prédite.
# 
# L'apprentissage de l'arbre s'effectue de manière récursive top-down : à chaque noeud, l'algorithme choisit le split vertical (seuillage
# d'une variable) qui optimise une mesure d'homogénéité sur la partition obtenue (usuellement l'[entropie de shanon](http://fr.wikipedia.org/wiki/Entropie_de_Shannon#D.C3.A9finition_formelle) ou l'[index de Gini](http://fr.wikipedia.org/wiki/Coefficient_de_Gini) : l'entropie d'une partition est d'autant plus petite qu'une classe prédomine dans chaque sous-
# ensemble de la partition, elle est nulle lorsque la séparation est parfaite).
# 
# Bien que l'algorithme pourrait continuer récursivement jusqu'à n'obtenir que des feuilles contenant un ensemble pur d'exemples (d'une seule classe), on utilise souvent des critères d'arrêts (pourquoi ? - nous y reviendrons lors de ce TP). Les plus communs sont les suivants :
# 
# + le nombre d'exemples minimum que doit contenir un noeud
# 
# + la profondeur maximale de l'arbre
# 
# + la différence de gain de la mesure d'homogénéité entre le noeud père et les noeuds fils
# 
# 
# 
# 

# ## Prise en main sklearn, données artificielles
# scikit-learn est un des modules de machine learning les plus populaires (installation : pip install scikit-learn --user).
# Il contient les algos que nous avons déjà vu (knn, noyaux, perceptron, regression), et bien d'autres outils et algorithmes.

# In[ ]:

from tools import * 
import numpy as np # module pour les outils mathématiques
import matplotlib.pyplot as plt # module pour les outils graphiques
import tools # module fourni en TP1
from sklearn import tree # module pour les arbres
from sklearn import ensemble # module pour les forets
from sklearn import cross_validation as cv
from sklearn import linear_model
from sklearn import neighbors
from IPython.display import Image
#import pydot

# Tous les modeles d'apprentissage sous scikit fonctionnent de la manière suivante :
# 
# + création du classifieur (ici  **cls=Classifier()**)
# 
# + réglage des paramètres (par exemple la profondeur maximale, le nombre d'exemples par noeud)
# 
# + apprentissage du classifieur par l'intermédiaire de la fonction **cls.fit(data,labels)** 
# 
# + prediction pour de nouveaux exemples : fonction **cls.predict(data)**
# 
# + score du classifieur (précision, pourcentage d'exemples bien classés) : fonction **cls.score(data,labels)**
# 
# Pour un arbre de déciion, la classe est **tree.DecisionTreeClassfier()**.
# Dans le cas des arbres de décisions, nous avons aussi la possibilité d'obtenir l'importance des variables, un score qui est d'autant plus grand que la variable est "utile" pour la classification.

# In[ ]:

#Initialisation
data,y=tools.gen_arti(2)
mytree=tree.DecisionTreeClassifier() #creation d'un arbre de decision
mytree.max_depth=10 #profondeur maximale de 5
mytree.min_samples_split=3 #nombre minimal d'exemples dans une feuille
#Apprentissage
mytree.fit(data,y)

#prediction
pred=mytree.predict(data)
#print "precision : ", 1.*(1.*pred!=y).sum()/len(y)

#ou directement pour la precision : 
#print "precision (score) : "  +` mytree.score(data,y)`

#Importance des variables :
#plt.subplot(1,2,2)
#plt.bar([1,2],mytree.feature_importances_)
#plt.title("Importance Variable")
#plt.xticks([1,2],["x1","x2"])

#Affichage de l'arbre
with open("mytree.dot","wb") as f:
    tree.export_graphviz(mytree,f)
    
#plt.subplot(1,2,1)
#plot_frontiere(data,mytree.predict)
#plt.title("Frontieres de decision")

# Sur différents jeux de données artificielles (des tps précédents) : 
# 
# + *<font style="BACKGROUND-COLOR: lightgray" color='red'>observer les frontières de décision en fonction de la taille de l'arbe.</font>*
# 
# + *<font style="BACKGROUND-COLOR: lightgray" color='red'>faites varier les différents paramètres disponibles (hauteur de l'arbre, nombre d'exemples dans les noeuds par exemple) et tracer la précision en fonction de ces paramètres.
# Que remarquez vous sur la précision ?</font>*
# 
# + *<font style="BACKGROUND-COLOR: lightgray" color='red'>Est-ce que cette valeur de précision vous semble une estimation fiable de l'erreur ? Pourquoi ?</font>*
# 
# ## Validation croisée : sélection de modèle
# 
# Il est rare de disposer en pratique d'un ensemble de test (on préfère inclure le plus grand
# nombre de données dans l'ensemble d'apprentissage). Pour sélectionner un modèle tout en considérant le plus grand nombre d'exemples possible pour l'apprentissage, on utilise généralement
# une procédure dite de sélection par validation croisée. Pour chaque paramètrisation du problème,
# une estimation de l'erreur empirique du classifieur appris est faîte selon la procédure suivante :
# 
# + l'ensemble d'apprentissage $E_{app}$ est partitioné en $n$ ensembles d'apprentissage $\{E_i\}$
# 
# + Pour $i=1..n$
# 
#   + l'arbre est appris sur $E_{app}$\ $E_i$
# 
#   + l'erreur en test $err(E_i)$ est évaluée sur $E_i$ (qui n'a pas servi à l'apprentissage à cette itération)
# 
# + l'erreur moyenne $err=\frac{1}{n}\sum_{i=1}^n err(E_i)$ est calculée, le modèle sélectionné est celui qui minimise cette erreur
# 
# 
# Ci-dessous quelques fonctions utiles pour la sélection de modèle :

# In[ ]:

#permet de partager un ensemble en deux ensembles d'apprentissage et de test 
data_train,data_test,y_train,y_test=cv.train_test_split(data,y,test_size=0.3)
mytree.fit(data_train,y_train)
#print "precision en test (split 30 %) : ", mytree.score(data_test,y_test)

#permet d'executer une n-validation croisée et d'obtenir le score pour chaque tentative
cross_val = cv.cross_val_score(mytree,data,y,cv=10)
#print "precision en test (10-fold validation) : ",cross_val
#print "moyenne : ",np.mean(cross_val)," (",np.std(cross_val),")"

#alternative : obtenir les indices et itérer dessus  
kf= cv.KFold(y.size,n_folds=10)
res_train=[]
res_test=[]
for cvtrain,cvtest in kf:
    mytree.fit(data[cvtrain],y[cvtrain])
    res_train+=[mytree.score(data[cvtrain],y[cvtrain])]
    res_test+=[mytree.score(data[cvtest],y[cvtest])]
#print "ou de maniere analogue : "
#print "precision en train : ",res_train
#print "precision en test : ",res_test
#print "moyenne train : ",np.mean(res_train)," (", np.std(res_train),")"             
#print "moyenne test : ",np.mean(res_test)," (",np.std(res_test),")"
             
    


# *<font style="BACKGROUND-COLOR: lightgray" color='red'>Manipuler sur les différents types de génération artificielle ces fonctions afin de trouver les meilleurs paramètres selon le problème. Tracer l'erreur d'apprentissage et l'erreur de test en fonction des paramètres étudiés. Que se passe-t-il pour des profondeurs trop élevées des arbres ?</font>*

# ## Classification données USPS
# 
# Tester sur les données USPS (en sélectionnant quelques sous-classes). Observer l'importance des variables. Afficher la matrice 2D de la variable importance de chaque pixel de l'image (avec **plt.imshow(matrix)**). Les résultats semble-t-ils cohérents ? 
# Utiliser l'algorithme du perceptron fourni par sklearn (**linear_model.Perceptron**) ou le votre et comparer les résultats obtenus pour les poids.
# 

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

idx1 = np.where(yTrain==1)
idx2 = np.where(yTrain==2)
idx1and2 = np.hstack((idx1,idx2))
dataTrain_biclass = dataTrain[idx1and2][0,:,:]
yTrain_biclass = yTrain[idx1and2][0,:]

idx1 = np.where(yTest==1)
idx2 = np.where(yTest==2)
idx1and2 = np.hstack((idx1,idx2))
dataTest_biclass = dataTest[idx1and2][0,:,:]
yTest_biclass = yTest[idx1and2][0,:]

###### Arbre de décision

uspsTree = tree.DecisionTreeClassifier() #creation d'un arbre de decision
uspsTree.max_depth=5 #profondeur maximale de 5
uspsTree.min_samples_split=1 #nombre minimal d'exemples dans une feuille

#Apprentissage
uspsTree.fit(dataTrain_biclass,yTrain_biclass)

#precision : 
#print "precision Tree (score) : "  +` uspsTree.score(dataTest_biclass,yTest_biclass)`

#Importance des variables :
#plt.subplot(1,2,1)
#plot(uspsTree.feature_importances_)
#plt.title("Importance Variable of Tree")

######## Perceptron
uspsPerceptron = linear_model.Perceptron()

#Apprentissage
uspsPerceptron.fit(dataTrain_biclass,yTrain_biclass)

#precision : 
#print "precision Perceptron (score) : "  +` uspsPerceptron.score(dataTest_biclass,yTest_biclass)`

#Importance des variables :
#plt.subplot(1,2,2)
#a = uspsPerceptron.coef_
#plot(uspsPerceptron.coef_)
#plt.title("Importance Variable of Perceptron")

###Validation croisée

#Concacténation des données d'apprentissage et de test
dataTotal_biclass = np.vstack((dataTest_biclass,dataTrain_biclass))
yTotal_biclass = np.hstack((yTest_biclass,yTrain_biclass))

#Arbre de décision
maxMeanScore = 0
argmax = [0,0]
for i in range(1,20,1):
    print i
    for j in range(1,10,1):
        USPSTreeBiclass = tree.DecisionTreeClassifier() #creation d'un arbre de decision
        USPSTreeBiclass.max_depth=i #profondeur maximale
        USPSTreeBiclass.min_samples_split=j #nombre minimal d'exemples dans une feuille
        cross_val = cv.cross_val_score(USPSTreeBiclass,dataTotal_biclass,yTotal_biclass,cv=10)
        if cross_val.mean()>maxMeanScore:
            maxMeanScore = cross_val.mean()
            argmax = [i,j]

print "max precision 10-fold validation for tree: ",maxMeanScore
print "obtained with [max_depthmin_samples_split]:,", argmax

#0.989 (max_depth=5, min_samples_split=1)
#0.990 (max_depth=5, min_samples_split=2)
#max precision 10-fold validation for tree:  0.992272727273
#obtained with [max_depthmin_samples_split]:, [7, 5]

#Perceptron
maxMeanScore = 0
argmax = 0
for k in range(-3,8,1):
    K = 10**(-k)
    print K
    USPSPerceptronBiclass = linear_model.Perceptron(penalty='l1',alpha=K)
    cross_val = cv.cross_val_score(USPSKnnBiclass,dataTotal_biclass,yTotal_biclass,cv=10)
    if cross_val.mean()>=maxMeanScore:
            maxMeanScore = cross_val.mean()
            argmax = K
            
print "max precision 10-fold validation for Perceptron: ",maxMeanScore
print "obtained with penalization coeff:,", argmax

#max precision 10-fold validation for Perceptron:  0.992272727273

#Knn
maxMeanScore = 0
argmax = 0
for k in range(1,20,1):
    print k
    USPSKnnBiclass = neighbors.KNeighborsClassifier(k)
    cross_val = cv.cross_val_score(USPSKnnBiclass,dataTotal_biclass,yTotal_biclass,cv=10)
    if cross_val.mean()>maxMeanScore:
            maxMeanScore = cross_val.mean()
            argmax = k

print "max precision 10-fold validation for Knn: ",maxMeanScore
print "obtained with number of neighbours:,", argmax
#max precision 10-fold validation for Knn:  0.996363636364
#obtained with number of neighbours:, 3 


###fôrets aléatoires

#forest = ensemble.RandomForestClassifier()
#forest.max_depth=7 #profondeur maximale
#forest.min_samples_split=2
#forest.fit(dataTrain_biclass,yTrain_biclass)
#forest.score(dataTest_biclass,yTest_biclass)

# forest.score(dataTest_biclass,yTest_biclass) = 0.985 (max_depth=7,min_samples_split=2)

maxMeanScore = 0
argmax = [0,0,0]

for k in range(15,30,1):
    print k
    for i in range(1,20,1):
        for j in range(1,10,1):
            forest = ensemble.RandomForestClassifier(k)
            forest.max_depth=i #profondeur maximale
            forest.min_samples_split=j #nombre minimal d'exemples dans une feuille
            forest.fit(dataTrain_biclass,yTrain_biclass)
            score = forest.score(dataTest_biclass,yTest_biclass)
            if score>maxMeanScore:
                maxMeanScore = score
                argmax = [k,i,j]
                #cross_val = cv.cross_val_score(forest,dataTotal_biclass,yTotal_biclass,cv=10)
                #if cross_val.mean()>maxMeanScore:
                #    maxMeanScore = cross_val.mean()
                #    argmax = [i,j]
        
print "max precision for Forest: ",maxMeanScore
print "obtained with [n_estimators,max_depth,min_samples_split]:,", argmax
#max precision for Forest:  0.993506493506
#obtained with [max_depthmin_samples_split]:, [8, 3]

#max precision 10-fold validation for Forest:  0.997727272727
#obtained with [max_depthmin_samples_split]:, [4, 1]

# Sur quelques exemples, comparer les performances des arbres, des knns (**neighbors.KNeighborsClassifier**) et du Perceptron en utilisant la validation croisée pour calibrer au mieux vos modèles. 
# 
# Expérimenter également les fôrets aléatoires : c'est une méthode de baging très utilisée, qui consiste à considérer un ensemble d'arbres appris chacun sur un échantillonage aléatoire de la base d'exemples; la classification se fait par vote majoritaire (**enemble.RandomForestClassifier()**).
# 
# 
# ## Classification sur la base movielens 
# 
# ### Introduction
# 
# La base movielens est une base de données issue d'imdb, qui contient des informations sur des films (le genre, l'année de production, des tags) et des notes attribuées par les utilisateurs. Elle est utilisée généralement pour la recommendation de films. Nous allons l'utiliser dans le cadre de la classification, afin de prédire si un film est bon ou mauvais, dans deux contextes :
# 
# + en prenant en compte uniquement l'information sur le film et le score moyen du film
# 
# + en prenant en compte l'information de l'utilisateur qui score le film
# 
# Télécharger l'[archive suivante](http://www-connex.lip6.fr/~baskiotisn/Telecom/mvlens.zip)
# 
# Le bloc de code suivant est utilisé pour  charger et prétraiter les données.
# 

# In[ ]:

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy import random
from collections import Counter

### liste des professions
occupation=[ "other", "academic/educator", "artist", "clerical/admin", "college/grad student", "customer service", "doctor/health care",
"executive/managerial", "farmer", "homemaker", "K-12 student", "lawyer", "programmer", "retired", "sales/marketing", "scientist",
"self-employed", "technician/engineer", "tradesman/craftsman", "unemployed", "writer"]

### liste des genres
genre=['unknown','Action','Adventure','Animation',"Children's",'Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir' ,
'Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']

dicGenre=dict(zip(genre,range(len(genre))))

dicOccup=dict(zip(occupation,range(len(occupation))))


def read_mlens(fname,sep='::'):
    """ Read a generic .dat file
            - fname : file name
            - sep : separator
        return : list of lists, each list contains the fields of each line read """

    def toint(x):
        if x.isdigit():
            return int(x)
        return x
    f=open(fname,'r')
    tmp= [ s.lstrip().rstrip().split(sep) for s in f.readlines()]
    #lstrip et rstrip enleve les blancs de debut et de fin
    f.close()
    return [ [toint(x) for x in y] for y in tmp ]
def read_movies(fname="movies.dat"):
    """ Read movies information, reindexing movies
        return : res: binary matrix nbmovies x genre, res[i,j]= 1 if movie i has the genre j, 0 otherwise
                 movies2id : original idMovie to reindexed id
                 dic2Movies : reindexed id to list of movie information (id, title, genre)
    """
    movies=read_mlens(fname)
    dicMovies=dict()
    movies2id=dict()
    res=np.zeros((len(movies),len(genre)))
    for i,m in enumerate(movies):
        dicMovies[i]=m
        movies2id[m[0]]=i
        for g in m[2].split('|'):
            res[i,dicGenre[g]]=1
    return res,movies2id,dicMovies

def read_users(fname="users.dat"):
    """ Read users informations
        return : nbusers * 3 : gender (1 for M, 2 for F), age, occupation index"""
    users=read_mlens(fname)
    res=np.zeros((len(users),3))
    for u in users:
        res[u[0]-1,:]=[u[1]=='M' and 1 or 2, u[2],int(u[3])]
    return res

def read_tags(fname="tags.dat"):
    """ Read tags informations
        return : dictionary : idTag->(label,number of uses)
        """
    tags=read_mlens(fname,"\t")
    res=dict()
    for t in tags:
        res[t[0]]=(t[1],t[2])
    return res


def read_files(mname="movies.dat",uname="users.dat",rname="ratings.dat",tname="tags.dat",trname="tag_relevance.dat"):
    """ Read all files
        return :
            * movies: binary matrix movies x genre
            * users : matrix users x (gender, age, occupation index)
            * ratings : matrix movies x users, with score 1 to 5
            * movies2id : dictionary original id to reindexed id
            * dicMovies : dictionary reindexed id to movie information
            * tags : dictionary idTag -> (name,popularity)
            * tagrelevance : matrix movies x tags, relevance
    """
    print "Reading movies..."
    movies,movies2id,dicMovies=read_movies(mname)
    print "Reading users..."
    users=read_users(uname)
    print "Reading ratings..."
    rtmp=read_mlens(rname)
    ratings=np.zeros((movies.shape[0],users.shape[0]))
    for l in rtmp:
        ratings[movies2id[l[1]],l[0]-1]=l[2]
    print "Reading tags..."
    tags=read_tags(tname)
    tagrelevance=np.zeros((movies.shape[0],len(tags)))
    print "Reading tags relevance..."
    with open(trname) as f:
        for i,ltag in enumerate(f):
            if i % 100000 == 0:
                print str(i /100000)+" 00k lines"
            ltag=ltag.rstrip().split("\t")
            if int(ltag[0]) in movies2id:
                tagrelevance[movies2id[int(ltag[0])],ltag[1]]=float(ltag[2])

    return movies,users,ratings,tags,tagrelevance,movies2id,dicMovies


###lire les fichiers
movies,users,ratings,tags,tagrelevance,movies2id,dicMovies=read_files()



# Les informations suivantes sont stockées :
# 
# + movies: une matrice binaire, chaque ligne un film, chaque colonne un genre, 1 indique le genre s'applique au film
# 
# + users : une matrice, chaque ligne un utilisateur, et les colonnes suivantes : sexe (1 masculin, 2 feminin),  age, index de la profession
# 
# + ratings : une matrice de score, chaque ligne un film, chaque colonne un utilisateur
# 
# + movies2id : dictionnaire permettant de faire la correspondance entre l'identifiant du film à l'identifiant réindexé
# 
# + dicMovies : dictionnaire inverse du précédent
# 
# + tags : dictionnaire des tags, identifiant vers le couple (nom,popularité)
# 
# + tagrelevance : matrice, chaque ligne un film, chaque colonne un tag, chaque case un score entre 0 et 1
# 
# ### Classification à partir de l'information unique du film
# 
# Notre matrice **movies** ne contient que les informations du genre du film. Il vous faut  y ajouter également l'année de production du film. Nous allons considérer le problème de classification binaire si un film est bon ou non, en considérant comme bon les filmes de score supérieur à 3. 
# 
# + *<font style="BACKGROUND-COLOR: lightgray" color='red'> Transformer les données et fabriquer les labels. Utiliser les arbres de décisions et le perceptron pour cette tache d'apprentissage.</font>*
# 
# + *<font style="BACKGROUND-COLOR: lightgray" color='red'> Sur quelques paramètres, que remarquez vous sur l'erreur d'apprentissage et de test ?</font>*
# 
# + *<font style="BACKGROUND-COLOR: lightgray" color='red'> La taille de l'ensemble de test joue-t-elle un rôle ?</font>*
# 
# + *<font style="BACKGROUND-COLOR: lightgray" color='red'> Tracer les courbes de ces deux erreurs en fonction de la profondeur. Que remarquez vous ? Quels sont les meilleurs paramètres pour l'erreur en apprentissage et en test ?</font>*
# 
# + *<font style="BACKGROUND-COLOR: lightgray" color='red'> Quelles sont les variables les plus importantes ?  </font>*
# 

# 
# ### Classification avec les informations utilisateurs
# 
# Proposer une manipulation des données qui permettent d'intégrer les informations utilisateurs dans le processus de classification. Tester (meme questions que précedement).

# + *<font style="BACKGROUND-COLOR: lightgray" color='red'> Etudier comme précédement les erreurs en test et en apprentissage. Comparer ces erreurs si vous réduisez les utilisateurs par tranche d'age. Remarquez-vous des tranches d'age plus stable ?   </font>*
# 
# + *<font style="BACKGROUND-COLOR: lightgray" color='red'> Comparer vos résultats si vous utilisez une fôret aléatoire (aggrégation d'arbres).</font>*
# 

###Transformation des données et fabrication des labels
def getYear(string):
    nbOccurence = string.count('(')
    for i in range(1,nbOccurence):
        string = string.split('(',1)[1]
    return int(string.split('(',1)[1].split(')')[0])

#Données
yearsOfProduction = np.zeros(movies.shape[0])
for i in range(yearsOfProduction.shape[0]):
    yearsOfProduction[i] = getYear(dicMovies[i][1])
yearsOfProduction = (yearsOfProduction - yearsOfProduction.mean())/(yearsOfProduction.max()-yearsOfProduction.min())

#Dictionnaire nom du tag -> indice du tag
tagsIndex = {}
for key in tags:
    tagsIndex[tags[key][0]]=(key,tags[key][1])
    
tagsOfMovies = np.zeros([movies.shape[0],len(tags)])
for i in range(tagsOfMovies.shape[0]):
     for tag in dicMovies[i][2].split('|'):
         if tag=="Children's":
             tag = "children"
         elif tag == "Film-Noir":
             tag = "film noir"
         if (tag in tagsIndex or tag.lower() in tagsIndex):
             tagsOfMovies[i,tagsIndex[tag.lower()][0]]=1./tagsIndex[tag.lower()][1]
         else:
             print tag

moviesNormalized = movies/movies.sum(0)
moviesNormalized[:,0]=0
#dataMovies = np.hstack((to_col(yearsOfProduction),tagsOfMovies))
dataMovies = to_col(yearsOfProduction)
#dataMovies = np.hstack((movies,to_col(yearsOfProduction)))
#labels
average = ratings.sum(1).sum(0)/len(np.where(ratings!=0)[1])
yMovies = ((ratings.sum(1)/(ratings!=0).sum(1))>3.28).astype(int)

### Perceptron
MoviePerceptron = linear_model.Perceptron()
kf= cv.KFold(yMovies.size,n_folds=10)
res_train=[]
res_test=[]
for cvtrain,cvtest in kf:
    MoviePerceptron.fit(dataMovies[cvtrain],yMovies[cvtrain])
    res_train+=[MoviePerceptron.score(dataMovies[cvtrain],yMovies[cvtrain])]
    res_test+=[MoviePerceptron.score(dataMovies[cvtest],yMovies[cvtest])]
print "moyenne et écart type train Perceptron: ",np.mean(res_train)," (", np.std(res_train),")"             
print "moyenne et écart type test Perceptron: ",np.mean(res_test)," (",np.std(res_test),")"

### Arbre de décision
maxMeanScoreTrain = [0,0]
maxMeanScoreTest = [0,0]
argmaxMaxTrain = [0,0]
argmaxMaxTest = [0,0]
for i in range(1,10,1):
        print i
        for j in range(1,10,1):
            kf= cv.KFold(yMovies.size,n_folds=10)
            res_train=[]
            res_test=[]
            for cvtrain,cvtest in kf:
                MovieTree = tree.DecisionTreeClassifier() #creation d'un arbre de decision
                #MovieTree = ensemble.RandomForestClassifier(500)
                MovieTree.max_depth=i #profondeur maximale
                MovieTree.min_samples_split=j 
                MovieTree.fit(dataMovies[cvtrain],yMovies[cvtrain])
                res_train+=[MovieTree.score(dataMovies[cvtrain],yMovies[cvtrain])]
                res_test+=[MovieTree.score(dataMovies[cvtest],yMovies[cvtest])]
            if np.mean(res_train)>maxMeanScoreTrain[0]:
                maxMeanScoreTrain = [np.mean(res_train),np.mean(res_test)]
                argmaxMaxTrain = [i,j]
            if np.mean(res_test)>maxMeanScoreTest[0]:
                maxMeanScoreTest = [np.mean(res_train),np.mean(res_test)]
                argmaxMaxTest = [i,j]
            
print "moyenne pour max Train:", maxMeanScoreTrain
print "obtained with [max_depth,min_samples_split]:", argmaxMaxTrain
print "moyenne pour max Test:", maxMeanScoreTest
print "obtained with [max_depth,min_samples_split]:", argmaxMaxTest

#Importance des variables :
MovieTree = tree.DecisionTreeClassifier(max_depth=8,min_samples_split=1)
MovieTree.fit(dataMovies,yMovies)
plt.figure(3)
plt.plot(MovieTree.feature_importances_)
plt.title("Importance Variable of Tree")

np.argsort(-MovieTree.feature_importances_)

##### movies (non normalized)+ years of production
#moyenne pour max Train: [0.80705080362575354, 0.65438806879919442]
#obtained with [max_depth,min_samples_split]: [19, 2]
#moyenne pour max Test: [0.7303631895429975, 0.69482150902393136]
#obtained with [max_depth,min_samples_split]: [8, 1]

##### movies ( normalized)+ years of production
#moyenne pour max Train: [0.80705080362575354, 0.65438806879919442]
#obtained with [max_depth,min_samples_split]: [19, 2]
#moyenne pour max Test: [0.7303631895429975, 0.69482150902393136]
#obtained with [max_depth,min_samples_split]: [8, 1]

##### labels normalized
#moyenne pour max Train: [0.70395172431300579, 0.6775773195876289]
#obtained with [max_depth,min_samples_split]: [9, 2]
#moyenne pour max Test: [0.7001747119320838, 0.68530729070044794]
#obtained with [max_depth,min_samples_split]: [8, 1]

##### years normalized
#moyenne pour max Train: [0.65627963080793328, 0.63639718548750435]
#obtained with [max_depth,min_samples_split]: [6, 1]
#moyenne pour max Test: [0.65516361176691207, 0.65542760978453862]
#obtained with [max_depth,min_samples_split]: [2, 9]

##### Classification avec les informations utilisateurs
Occup = np.zeros([movies.shape[0],len(dicOccup)])
Sex = np.zeros(movies.shape[0])
Age = np.zeros(movies.shape[0])
AgeClass = np.zeros([movies.shape[0],len(Counter(users[:,1]))])
AgeDicIndex = {}
for index,age in enumerate(sorted(Counter(users[:,1]).keys())):
    AgeDicIndex[age]=index

for i,x in enumerate(ratings):
    index = np.where(x!=0)[0]

    if len(index)==0:
        pass
    else:
        Sex[i] = (users[index,0].sum()-len(index))/len(index)
        Age[i] = users[index,1].sum()/len(index)
        dicAge = Counter(users[index,1])
        for key in dicAge:
            AgeClass[i,AgeDicIndex[key]] = float(dicAge[key])/sum(dicAge.values())
        dicOccup = Counter(users[index,2])
        for key in dicOccup:
            Occup[i,int(key)] = dicOccup[key]
    if i==0:
        pass

dataMovies = np.hstack((moviesNormalized,to_col(yearsOfProduction),Occup,to_col(Sex),AgeClass))

dataMovies = to_col(Sex)
dataMovies = to_col(Age)
dataMovies = AgeClass
dataMovies = to_col(tagsOfMovies)
dataMovies = to_col(np.hstack((np.ones(movies.shape[0]/2),np.zeros(movies.shape[0]-movies.shape[0]/2))))
### X = occup + Age + Sex
#moyenne pour max Train: [0.99230270080817073, 0.74066400763257634]
#obtained with [max_depth,min_samples_split]: [19, 1]
#moyenne pour max Test: [0.78435868396507236, 0.76641467680809905]
#obtained with [max_depth,min_samples_split]: [4, 1]

### X = genre + année + occup + Age + Sex
#moyenne pour max Train: [0.99510676385350538, 0.75507977102271229]
#obtained with [max_depth,min_samples_split]: [19, 1]
#moyenne pour max Test: [0.81314521603763024, 0.78728235231759991]
#obtained with [max_depth,min_samples_split]: [5, 1]