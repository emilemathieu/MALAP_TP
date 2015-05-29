# -*- coding: utf-8 -*-

from tools import * 
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
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

#Lena_RGB=Load_Lena()
#KMeans_RGB=KMeans(32,n_init=5,verbose=1)
#KMeans_RGB.fit(Lena_RGB)

#Lena_64[:,:]=KMeans_RGB.cluster_centers_[KMeans_RGB.labels_,:]
#mnew=Lena_64.reshape((512,512,3))
#plt.imshow(mnew)

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
#dataMovies = to_col(yearsOfProduction)
#dataMovies = np.hstack((movies,to_col(yearsOfProduction)))
#labels

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

average = ratings.sum(1).sum(0)/len(np.where(ratings!=0)[1])
yMovies = ((ratings.sum(1)/(ratings!=0).sum(1))>3.28).astype(int)
        
dataMovies=np.hstack((to_col(yearsOfProduction),moviesNormalized))
#dataMovies=np.hstack((to_col(Sex),AgeClass))

### Calcul matrice de similarité

def Similarity_Matrix(data):
    M=np.zeros([dataMovies.shape[0],dataMovies.shape[0]])
    for i,x in enumerate(data):
        for j,y in enumerate(data): 
            M[i,j]=np.exp(-(np.inner(x-y,x-y)))
            
M=Similarity_Matrix(dataMovies)
plt.imshow(M)

# Nombre de clusters
k=3

def Spectral_Clustering(k,M):
    MovieSpectralClustering = SpectralClustering(k)
    MovieSpectralClustering.affinity=="precomputed"
    MovieSpectralClustering.fit(M)
    IndexSpectral=np.argsort(MovieSpectralClustering.labels_)
    Mcluster = M[IndexSpectral,:]
    Mcluster = Mcluster[:,IndexSpectral]
    plt.imshow(Mcluster)
    
Spectral_Clustering(k,M)

def K_Means(k,data):
    MovieKMeans = KMeans(k)
    IndexKMeans=np.argsort(MovieKMeans.fit_predict(data))
    MKMeans = M[IndexKMeans,:]
    MKMeans = MKMeans[:,IndexKMeans]
    plt.imshow(MKMeans)
    
K_Means(k,dataMovies)