# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# #Premiers algorithmes : régression linéaire, perceptron 
# L'objectif de ce tp est de comprendre en profondeur l'algorithme de la descente du gradient, algorithme d'optimisation de fonction convexe. La première partie de ce tp est dédiée à la prise en main de python et à l'étude de cet algorithme dans le cas de fonctions simples. La deuxième partie est consacrée à l'application de cet algorithme au cas de la regression linéaire. La troisième partie étudie son application dans le cas de l'algorithme du perceptron, algorithme de classification. Enfin, la quatrième partie étudie quelques extensions, de la ridge regression à une introduction aux svms. 

# <codecell>

from tools import *
plt.ion()
#matplotlib inline
#import numpy as np
#from matplotlib import pyplot as plt
# <markdowncell>

# ## Descente de gradient
# 
# ###Préambule
# Télécharger le fichier [tools.py](http://). Un fichier python est comme un module : vous pouvez charger tout ce que contient un fichier en exécutant `from tools import *` . De manière générale, nous aurons besoin des modules `numpy` (pour les maths), `matplotlib.pyplot` (pour les courbes et graphiques) et `sklearn` (pour l'apprentissage). Vous pouvez donner un racourci à un module (exemple classique : `import numpy as np`, puis `np.fonction` pour appeller une fonction du module `numpy`).  
# 
# ###Algorithme
# L'algorithme de descente du gradient est un algorithme itératif très utilisé pour optimiser une fonction continue dérivable. Son principe est d'approché pas à pas une solution (localement) optimale, en "suivant" la direction du gradient. A partir d'un point tiré aléatoirement $x_0$, le point est mis à jour itérativement en se déplaçant en direction inverse du gradient de la fonction $f$ :
# 
# 1. $x_0=random()$
# 2. $x_{i+1} \leftarrow x_i -\epsilon*\nabla f(x_i)$
# 3. boucler sur 2.
# 
# La classe `OptimFunc` permet d'enregistrer les renseignements nécessaires à l'optimisation d'une fonction $f$ : la fonction elle-même, son gradient, et la dimension de l'entrée. Un exemple est donné ci-dessous dans le cas de la fonction $f(x)$.

# <codecell>

class OptimFunc:
    def __init__(self,f=None,grad_f=None,dim=2):
        self.f=f
        self.grad_f=grad_f
        self.dim=dim
    def init(self,low=-5,high=5):
        return random.random(self.dim)*(high-low)+low

def lin_f(x): return x
def lin_grad(x): return 1
lin_optim=OptimFunc(lin_f,lin_grad,1)
#Utiliser la fonction :
lin_optim.f(3)
#le gradient :
lin_optim.grad_f(1)

# <markdowncell>

# #### Coder les fonctions suivantes et les instances de `OptimFunc` `xcosx` et `rosen` qui y correspondent :
# 
# - $xcosx(x)=x cos(x)$ en dimension 1
def xcosx_f(x): return np.cos(x)*x
def xcosx_grad(x): return -np.sin(x)*x+np.cos(x)
xcosx=OptimFunc(xcosx_f,xcosx_grad,1)
# - $rosen(x_1,x_2)=100*(x_2-x_1^2)^2+(1-x_1)^2$  en dimension 2
def rosen_f(x): 
    x = to_line(x)
    return 100*((x[:,1]-x[:,0]**2))**2+(1-x[:,0])**2
def rosen_grad(x): 
    x = to_line(x)
    return np.array([-400*x[:,0]*(x[:,1]-x[:,0]**2)-2*(1-x[:,0]),200*(x[:,1]-x[:,0]**2)]).T
rosen=OptimFunc(rosen_f,rosen_grad,2)

#  
# #### Utiliser le code suivant (en le comprenant) pour afficher les fonctions précédentes.

# <codecell>

xrange=np.arange(-5,5,0.1)
#plt.plot(xrange,xcosx.f(xrange))
#plt.show()

### affichage 3D
grid,xvec,yvec=make_grid(xmin=-5,xmax=5,ymin=-5,ymax=5)
#z=rosen.f(grid).reshape(xvec.shape)
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#surf = ax.plot_surface(xvec, yvec, z, rstride=1, cstride=1, cmap=cm.gist_rainbow,linewidth=0, antialiased=False)
#fig.colorbar(surf)
#plt.show()


# <markdowncell>

# #### Utiliser la classe suivante pour optimiser les fonctions précédentes.
# Cette classe implémente une descente de gradient. Quelle est le critère d'arrêt ? en voyez-vous d'autres ? A quoi correspond les variables `log_x`, `log_f`, `log_grad` ?

# <codecell>

class GradientDescent:
    def __init__(self,optim_f,eps=1e-4,max_iter=5000):
        self.optim_f=optim_f
        self.eps=eps
        self.max_iter=max_iter
    def reset(self):
        self.i=0
        self.w = self.optim_f.init()
        self.log_w=np.array(self.w)
        self.log_f=np.array(self.optim_f.f(self.w))
        self.log_grad=np.array(self.optim_f.grad_f(self.w))
    def optimize(self,reset=True):
        if reset:
            self.reset()
        while not self.stop():
            self.w = self.w - self.get_eps()*self.optim_f.grad_f(self.w)
            self.log_w=np.vstack((self.log_w,self.w))
            self.log_f=np.vstack((self.log_f,self.optim_f.f(self.w)))
            self.log_grad=np.vstack((self.log_grad,self.optim_f.grad_f(self.w)))
            if self.i%500==0:
                print self.i," iterations ",self.log_f[self.i]
            self.i+=1
    def stop(self):
        return (self.i>2) and (self.max_iter and (self.i>self.max_iter))
    def get_eps(self):
        return self.eps

#xcosx_descent = GradientDescent(xcosx,max_iter=15000)
#rosen_descent = GradientDescent(rosen,max_iter=15000)
#xcosx_descent.optimize()
#rosen_descent.optimize()

# <markdowncell>

# Tracez les courbes de la valeur de la fonction en fonction du nombre d'itérations en faisant varier la variable `epsilon`. Que remarquez-vous ? Tracer la trajectoire d'optimisation à l'aide des fonctions d'affichage précédentes.
# Trouvez-vous toujours la même solution en fonction des exécutions ? Si non, de quoi dépend-elle ? Est-ce normal ?

#plt.plot(xrange,xcosx_descent.optim_f.f(xrange))
#plt.plot(xcosx_descent.log_w,xcosx_descent.log_f,"+r")
#plt.show()

#grid,xvec,yvec=make_grid(xmin=-5,xmax=5,ymin=-5,ymax=5)
#z=rosen.f(grid).reshape(xvec.shape)
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#surf = ax.plot_surface(xvec, yvec, z, rstride=1, cstride=1, cmap=cm.gist_rainbow,linewidth=0, antialiased=False)
#fig.colorbar(surf)
#plt.plot(rosen_descent.log_w[:,0],rosen_descent.log_w[:,1],rosen_descent.log_f,"+r")
#plt.show()

# <markdowncell>

# ## Régression linéaire
# Etant donné un ensemble de $n$ points en $d$ dimensions et leurs valeurs cibles  $\{(x^i,y^i)\}\in X\times Y \subset \mathbb{R}^d\times \mathbb{R}$, le problème de la régression linéaire est de trouver une fonction linéaire $f : X \to Y$ qui minimise l'erreur quadratique $\sum_i \frac{1}{2n}(f(x^i)-y^i)^2$. La fonction $f$ étant linéaire, on peut la paramétriser par $\mathbf{w} \in \mathbb{R}^{d+1}$ : $f_\mathbf{w}(x)=\sum_{i=1}^d w_i x_i+w_0$. L'objectif est alors de trouver $\mathbf{w}$ qui minimise l'erreur quadratique. En cours vous avez vu la résolution exacte. Dans ce qui suit, nous allons étudier la résolution de ce problème à l'aide de la descente du gradient.
# ￼
# - Faites une fonction `gen_1d(n,eps)` qui engendre $n$ données 1D selon la droite $f(x)=2*x+1$ que vous bruiterez avec un bruit gaussien de variance `eps` (avec un code de deux lignes, en utilisant les fonctions `np.random.random()` et `np.random.normal()`, étudier l'aide avec la fonction `help()`). Cette fonction doit vous rendre les coordonnées $x$ des points ainsi que leurs valeurs.
# - Quel est le calcul matriciel permettant de calculer $f_\mathbf{w}$ ? Et son gradient ?
# - Quelle est la fonction (et son gradient) à optimiser dans ce cas ? Comment adapter la descente de gradient à ce problème ?
# - A quoi correspond $w_0$ ? Comment l'intégrer en pratique ?
# - Utiliser une classe héritée (voir squelette ci-dessous) pour coder la régression linéaire.
# - Tester votre fonction sur les données engendrées. Tracez la solution.

# <codecell>

def gen_1d(n,eps):
    #xrange=np.arange(0,n,1)
    xrange = np.random.random(n)
    vec = 2*xrange+1
    vec = vec + np.random.normal(0,eps,n)
    #np.vstack((xrange,vec)).T
    return xrange.T,vec.T
    
def gen_1dbis(n,eps):
    #xrange=np.arange(0,n,1)
    xrange = np.random.random(n)*10
    vec = 2*xrange+1+2*(np.multiply(xrange,xrange))
    vec = vec + np.random.normal(0,eps,n)
    #np.vstack((xrange,vec)).T
    return xrange.T,vec.T

test,testy = gen_1dbis(500,1)

def fw_f(w,x):
    return np.inner(w,x)
    #return np.sum(w*x)
def fw_grad(x): return w

#def phi(data,x,w,simga):
#    print(data.shape)
#    print(x.shape)
#    c = np.exp(np.linalg.norm(data-x))/simga
#    print(c.shape)
#    return c.dot(w)

#data1,y1 = gen_1d(500,0.1) 

#data1 = data1.reshape((500,1))
#print(data1.shape)
#print(np.ones((500,1)).shape)
#data1 = np.hstack((np.ones((500,1)),data1))

class Regression(Classifier,GradientDescent,OptimFunc):
    def __init__(self,eps=1e-7,max_iter=30000):
        GradientDescent.__init__(self,self,eps,max_iter)
        self.dim=self.data=self.y=self.n=self.w=None
    def fit(self,data,y):
        self.y=y
        self.n=y.shape[0]
        self.dim=data.size/self.n+1
        self.data=data.reshape((self.n,self.dim-1))
        self.data=np.hstack((np.ones((self.n,1)),self.data))
        self.optimize()
    def f(self,w):
        # ||XW-Y||2
        return np.linalg.norm(self.data.dot(w)-self.y)
        #return np.linalg.norm(phi(data1,self.data,w,1)-self.y)
    def grad_f(self,w):
        #2X'(XW-Y)
        return 2*self.data.T.dot((self.data.dot(w)-self.y))
        #return 2*self.data.T.dot(phi(data1,self.data,w,1)-self.y)
    def init(self):
        return np.random.random(self.dim)*(np.max(self.data)-np.min(self.data))+np.min(self.data)
    def predict(self,data):
        n=data.size/(self.dim-1)
        wx = np.hstack((np.ones((n,1)),data.reshape(n,self.dim-1))).dot(self.w)
        return wx

class Regression2(Classifier,GradientDescent,OptimFunc):
    def __init__(self,eps=1e-7,max_iter=30000):
        GradientDescent.__init__(self,self,eps,max_iter)
        self.dim=self.data=self.y=self.n=self.w=None
    def fit(self,data,y):
        self.y=y
        self.n=y.shape[0]
        self.dim=data.size/self.n+1
        self.data=data.reshape((self.n,self.dim-1))
        self.data=np.hstack((np.ones((self.n,1)),self.data))
        self.optimize()
    def f(self,w):
        # ||XW-Y||2
        return np.linalg.norm(self.data.dot(w)+(self.data.dot(w))**2-self.y)
        #return np.linalg.norm(phi(data1,self.data,w,1)-self.y)
    def grad_f(self,w):
        #2X'(XW-Y)
        return 2*self.data.T.dot((self.data.dot(w)+(self.data.dot(w))**2-self.y))
        #return 2*self.data.T.dot(phi(data1,self.data,w,1)-self.y)
    def init(self):
        return np.random.random(self.dim)*(np.max(self.data)-np.min(self.data))+np.min(self.data)
    def predict(self,data):
        n=data.size/(self.dim-1)
        wx = np.hstack((np.ones((n,1)),data.reshape(n,self.dim-1))).dot(self.w)
        return wx + np.multiply(wx,wx)

#data,y = gen_1dbis(1000,3)
#regression = Regression()
#regression.fit(data,y)
#dataForPrediction,yPrediction = gen_1dbis(1000,2)
#predicted = regression.predict(dataForPrediction)

#plt.scatter(data,y)
#plt.title('Regression Lineaire par descente du gradient')
#plt.xlabel('x')
#plt.xlabel('y')
#plt.plot(dataForPrediction,predicted,'+r')
#plt.show()

# <markdowncell>

# ## Premier algorithme de classification: Algorithme du perceptron
# Cet algorithme a une longue histoire et est la base des réseaux de neurones. Il peut être vu comme une descente du gradient sur un coût particulier (même si historiquement son inspiration est autre).
# 
# On se place dans le cadre de la classification binaire : on considère deux labels, $Y=\{-1/+1\}$, et un ensemble de données $\{(x^i,y^i)\}\in X\times Y \subset \mathbb{R}^d\times Y$. On cherche une fonction $f$ qui permette de **généraliser** l'ensemble des données et de faire le moins d'erreurs sur l'ensemble disponible. Nous nous plaçons toujours dans le cadre linéaire, et la classification est faite  en considérant le signe de $f(x)$ ($>0 \to +1, ~ < 0\to -1$).
# Le coût qui nous intéresse est l'erreur $0/1$, qui compte le pourcentage d'erreurs qui est fait sur l'ensemble des données : $l(f(x),y)=\sum_i \mathbf{1}_{f(x^i)\not = y^i}$. 
# 
# Cependant ce coût est difficile à optimiser : pourquoi ?  A la place de ce coût, on va chercher à optimiser un coût surrogate, appelé coût *perceptron* (ou plus général *hinge loss*) : $l(f(x),y)=\sum_i max(0,-y^i f(x^i))$.
# 
# - Que se passe-t-il pour le coût si une erreur est faite pour $x$ ? si pas d'erreur ? 
# - Que représente cette erreur d'un point de vue géométrique ? et $w$ ?
# - Pourquoi peut-on adapter l'algorithme précédent à ce contexte pour optimiser $f$ ?
# - Calculer le gradient de la fonction de coût et adapter la classe précédente pour la classification (classe `Perceptron`).
# - Utiliser la fonction de génération de données `gen_arti` pour engendrer des jeux de données (3 jeux de données disponibles). Tracer les frontières de décisions (`plot_frontiere(data,perceptron.predict)`). Quelles sont les limites de l'algorithme du Perceptron ?
# 
def sgn(x):
    res = np.array(x)
    for i in range(res.shape[0]):
        if (res[i]>0):
            res[i] = +1
        else:
            res[i] = -1
    return res

class Perceptron(Classifier,GradientDescent,OptimFunc):
    def __init__(self,eps=1e-4,max_iter=2000):
        GradientDescent.__init__(self,self,eps,max_iter)
        self.dim=self.data=self.y=self.n=self.w=None
    def fit(self,data,y):
        self.y=y
        self.n=y.shape[0]
        self.dim=data.size/self.n+1
        self.data=data.reshape((self.n,self.dim-1))
        self.data=np.hstack((np.ones((self.n,1)),self.data))
        self.optimize()
    def f(self,w):
        #max(O,-XW.*Y)
        m = -np.multiply(self.data.dot(w),self.y)
        return np.sum(np.maximum(0,m))
    def grad_f(self,w):
        #m = -np.multiply(self.data.dot(w),self.y)
        #g = np.multiply(np.maximum(0,m),-self.y)
        #return np.sum(g)*w
        return sum(-(self.data*to_col(self.y)))
    def init(self):
        return np.random.random(self.dim)*(np.max(self.data)-np.min(self.data))+np.min(self.data)
    def predict(self,data):
        n=data.size/(self.dim-1)
        return np.hstack((np.ones((n,1)),data.reshape(n,self.dim-1))).dot(self.w)

data,y=gen_arti(0,200)

perceptron = Perceptron(1e-4)
perceptron.fit(data,y)
plot_frontiere(data,perceptron.predict)
plot_data(data,y)
plt.show()
print perceptron.w

class PerceptronDany(Classifier,GradientDescent,OptimFunc):
    def __init__(self,eps=5e-3,max_iter=1000):
        GradientDescent.__init__(self,self,eps,max_iter)
        self.dim=self.data=self.y=self.n=self.w=None
    def fit(self,data,y):
        self.y=y
        self.n=y.shape[0]
        self.dim=data.size/self.n+1
        self.data=data.reshape((self.n,self.dim-1))
        self.data=np.hstack((np.ones((self.n,1)),self.data))
        self.optimize()
    def f(self,w):
        #print('w',w.shape)
        temp=np.dot(self.data,w.T)
        #print(temp,'fin')
        y=[]
        for i in range(self.n):
            #print('ite',i,temp[i])
            if (-temp[i]*self.y[i]>0):
                y.append(-temp[i]*self.y[i])
            else:
                y.append(0)
        #print(y)
        return sum(y)
    def grad_f(self,w):
        test=np.zeros(self.n)
        indicator=(test==self.f(w)).astype(int)
        indicator=(to_col(np.ones(self.n))-to_col(indicator))
        #print(indicator)
        #print(self.data)
        res=-indicator*(self.data*to_col(self.y))
        #print(res)
        return sum(res)
    def trace(self,data,y):
        plt.plot(data,y,'ro')
        plt.plot(data,self.predict(data),color='b')
        plt.show()
    def init(self):
        return np.random.random(self.dim)*(np.max(self.data)-np.min(self.data))+np.min(self.data)
    def predict(self,data):
        n=data.size/(self.dim-1)
        return np.sign(np.hstack((np.ones((n,1)),data.reshape(n,self.dim-1))).dot(self.w.T))


# ## Extensions : linéaire... vraiment ?
# 
# Nous avons pour l'instant considérer que des fonctions linéaires. 
# 
# - Pourquoi à votre avis est-il utile de limiter la famille de fonctions considérée ? Quelle est la limite ?
# - Une façon d'augmenter l'expressivité des fonctions est de transformer l'espace d'entrée par des projections. Soit $x \in \mathbb{R}^2$, et $\phi(x)=(1,x_1,x_2,x_1x_2,x_1^2,x_2^2)$ la projection polynomiale d'ordre 2. Quelle est la forme des frontières de décision de la fonction de décision $f^\phi_\mathbf{w}(x)=f_\mathbf{w}(\phi(x))$ ? 
# - Que doit-on changer pour adapter la descente du gradient, que ce soit dans le cas de la régression linéaire ou du perceptron ? Peut-on généraliser à des degrés supérieurs ?  
# - Soit $B=\{x_1,x_2,\cdots,x_B\}$ un ensemble de points de même dimension que l'entrée, et $\phi_B(x)=(k(x,x_1),k(x,x_2),\cdots,k(x,x_B))$ la projection gaussienne sur $B$ de $x$, $k(x,x')=Ke^{\frac{\|x-x'\|^2}{\sigma^2}}$. Que doit-on changer pour adapter l'algorithme du perceptron ? 
# - Que veut-dire un poids positif devant une composante  de cette projection ? un poids négatif  ? un poids nul ? 
# - Que se passe-t-il si beaucoup de poids sont non nuls ? nuls ? Dans quel cas la frontière est la plus complexe ? la moins ?
# - En vous inspirant de la notion de ridge regression vu en cours, que proposez vous pour régulariser ?
# 
# 
# 

# <codecell>


