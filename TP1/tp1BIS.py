# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# #Premiers algorithmes : régression linéaire, perceptron 
# L'objectif de ce tp est de comprendre en profondeur l'algorithme de la descente du gradient, algorithme d'optimisation de fonction convexe. La première partie de ce tp est dédiée à la prise en main de python et à l'étude de cet algorithme dans le cas de fonctions simples. La deuxième partie est consacrée à l'application de cet algorithme au cas de la regression linéaire. La troisième partie étudie son application dans le cas de l'algorithme du perceptron, algorithme de classification. Enfin, la quatrième partie étudie quelques extensions, de la ridge regression à une introduction aux svms. 

# <codecell>

from tools import *
plt.ion()

# ## Descente de gradient
# <codecell>

class OptimFunc:
    def __init__(self,f=None,grad_f=None,dim=2):
        self.f=f
        self.grad_f=grad_f
        self.dim=dim
    def init(self,low=-5,high=5):
        return 1
        #return random.random(self.dim)*(high-low)+low

# #### Fonctions et instances de `OptimFunc` `xcosx` et `rosen`:

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

# <codecell>

xrange=np.arange(-5,5,0.1)
def plot_xcosx():
    plt.plot(xrange,xcosx.f(xrange))
    plt.show()

def plot_rosen():
    grid,xvec,yvec=make_grid(xmin=-5,xmax=5,ymin=-5,ymax=5)
    z=rosen.f(grid).reshape(xvec.shape)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(xvec, yvec, z, rstride=1, cstride=1, cmap=cm.gist_rainbow,linewidth=0, antialiased=False)
    fig.colorbar(surf)
    plt.show()

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

def xcosx_descent():
    xcosx_descent = GradientDescent(xcosx,eps=0.6,max_iter=25000)
    xcosx_descent.optimize()
    plt.plot(xrange,xcosx_descent.optim_f.f(xrange))
    plt.plot(xcosx_descent.log_w,xcosx_descent.log_f,"+r")
    plt.show()
    plt.plot(xcosx_descent.log_f,"+r")
    plt.show()

def rosen_descent():
    rosen_descent = GradientDescent(rosen,max_iter=25000)
    rosen_descent.optimize()
    grid,xvec,yvec=make_grid(xmin=-5,xmax=5,ymin=-5,ymax=5)
    z=rosen.f(grid).reshape(xvec.shape)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(xvec, yvec, z, rstride=1, cstride=1, cmap=cm.gist_rainbow,linewidth=0, antialiased=False)
    fig.colorbar(surf)
    plt.plot(rosen_descent.log_w[:,0],rosen_descent.log_w[:,1],rosen_descent.log_f,"+r")
    plt.show()

# <codecell>
# ## Régression linéaire

def gen_1d(n,eps):
    xrange = np.random.random(n)
    vec = 2*xrange+1
    vec = vec + np.random.normal(0,eps,n)
    return xrange.T,vec.T
    
def gen_1d_quad(n,eps):
    xrange = np.random.random(n)*10
    vec = 2*xrange+1+2*(np.multiply(xrange,xrange))
    vec = vec + np.random.normal(0,eps,n)
    return xrange.T,vec.T

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
        return np.linalg.norm(self.data.dot(w)-self.y)**2
    def grad_f(self,w):
        #return 2*
        return 2*self.data.T.dot((self.data.dot(w)-self.y))
    def init(self):
        return np.random.random(self.dim)*(np.max(self.data)-np.min(self.data))+np.min(self.data)
    def predict(self,data):
        n=data.size/(self.dim-1)
        wx = np.hstack((np.ones((n,1)),data.reshape(n,self.dim-1))).dot(self.w)
        print self.w
        return wx

class RegressionQuadratic(Classifier,GradientDescent,OptimFunc):
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
        return np.linalg.norm(self.data.dot(w)+(self.data.dot(w))**2-self.y)
    def grad_f(self,w):
        return 2*self.data.T.dot((self.data.dot(w)+(self.data.dot(w))**2-self.y))
    def init(self):
        return np.random.random(self.dim)*(np.max(self.data)-np.min(self.data))+np.min(self.data)
    def predict(self,data):
        n=data.size/(self.dim-1)
        wx = np.hstack((np.ones((n,1)),data.reshape(n,self.dim-1))).dot(self.w)
        print self.w
        return wx + np.multiply(wx,wx)

def LinearRegression():
    data,y = gen_1d(1000,1)
    regression = Regression(eps=1e-4)
    regression.fit(data,y)
    dataForPrediction,yPrediction = gen_1d(1000,1)
    predicted = regression.predict(dataForPrediction)

    plt.scatter(data,y)
    plt.title("Regression Lineaire par descente du gradient d'une fonction lineaire")
    plt.xlabel('x')
    plt.xlabel('y')
    plt.plot(dataForPrediction,predicted,'+r')
    plt.show()

def QuadraticRegression():
    data,y = gen_1d_quad(1000,3)
    regression = Regression()
    regression.fit(data,y)
    dataForPrediction,yPrediction = gen_1d_quad(1000,2)
    predicted = regression.predict(dataForPrediction)

    plt.scatter(data,y)
    plt.title("Regression Lineaire par descente du gradient d'une fonction quadratique")
    plt.xlabel('x')
    plt.xlabel('y')
    plt.plot(dataForPrediction,predicted,'+r')
    plt.show()
    
    data,y = gen_1d_quad(1000,3)
    regression = RegressionQuadratic()
    regression.fit(data,y)
    dataForPrediction,yPrediction = gen_1d_quad(1000,2)
    predicted = regression.predict(dataForPrediction)

    plt.scatter(data,y)
    plt.title("Regression quadratique par descente du gradient d'une fonction quadratique")
    plt.xlabel('x')
    plt.xlabel('y')
    plt.plot(dataForPrediction,predicted,'+r')
    plt.show()
  
# ## Premier algorithme de classification: Algorithme du perceptron

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
        m = -np.multiply(self.data.dot(w),self.y)
        return np.sum(np.maximum(0,m))
    def grad_f(self,w):
        return sum(-(self.data*to_col(self.y)))
    def init(self):
        return np.random.random(self.dim)*(np.max(self.data)-np.min(self.data))+np.min(self.data)
    def predict(self,data):
        n=data.size/(self.dim-1)
        return np.hstack((np.ones((n,1)),data.reshape(n,self.dim-1))).dot(self.w)

Bset=np.ones([3,2])
Bset[1,1]=-1
Bset[2,0]=-1
#Bset[3,0]=-1
#Bset[3,1]=-1
Bset=np.hstack((np.ones((Bset.shape[0],1)),Bset))

class PerceptronGaussian(Classifier,GradientDescent,OptimFunc):
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
        sigma = 1
        X = np.zeros([self.data.shape[0],Bset.shape[0]])
        for i,x in enumerate(Bset):
            #print np.sum(np.multiply(x-self.data,x-self.data),1).shape
            X[:,i]=np.exp(-np.sum(np.multiply(x-self.data,x-self.data),1)/sigma**2)
        #print X.shape
        m=-np.multiply(X.dot(w.T),self.y)
        #m = -np.multiply(self.data.dot(w),self.y)
        return np.sum(np.maximum(0,m))
    def grad_f(self,w):
        return sum(-(self.data*to_col(self.y)))
    def init(self):
        return np.random.random(self.dim)*(np.max(self.data)-np.min(self.data))+np.min(self.data)
    def predict(self,data):
        n=data.size/(self.dim-1)
        data=np.hstack((np.ones((n,1)),data.reshape(n,self.dim-1)))
        sigma = 1
        X = np.zeros([data.shape[0],Bset.shape[0]])
        for i,x in enumerate(Bset):
            X[:,i]=np.exp(-np.sum(np.multiply(x-data,x-data),1)/sigma**2)
        return X
        #return np.hstack((np.ones((n,1)),data.reshape(n,self.dim-1))).dot(self.w)

def Perceptron_example():
    data,y=gen_arti(1,200)
    perceptron = PerceptronGaussian(1e-4)
    perceptron.fit(data,y)
    plot_frontiere(data,perceptron.predict)
    plot_data(data,y)
    plt.show()
    print perceptron.w


# <codecell>
# ## Extensions : linéaire... vraiment ?

