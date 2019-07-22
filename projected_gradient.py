import numpy as np
import cvxopt
from cvxopt import matrix
import matplotlib.pyplot as plt

np.random.seed(777)
num_iter = 500
n = 200
d = 5
lamda = 1.0

#prepare dataset
omega = np.random.randn(d)
noise = 0.8 * np.random.randn()

K = np.zeros((n,n), dtype = int)

x = np.random.randn(n,d)
y = np.zeros(n,dtype = int)
for i in range(n):
    y[i] = 2 * (np.dot(omega,x[i]) + noise > 0) - 1
for i in range(n):
    for j in range(n):
        K[i,j] = y[i] * y[j] * np.dot(x[i],x[j])


zero = np.zeros(n, dtype = int)
one = np.ones(n, dtype = int)

#initial state
a = np.random.rand(n)


t = 0

#prepare cvxopt
I = matrix(np.identity(n))
c = np.zeros((2*n,n),dtype = float)
for i in range(n):
    c[i,i] = 1.0
    c[i+n,i] = -1.0
G = matrix(c)

d = np.zeros(2*n, dtype = float)
for i in range(n):
    d[i] = 1.0
h = matrix(d)

gap = []
primal = []
dual = []
for t in range(num_iter):
    eta = 0.5 / (t+1)
    b = a - eta * (1.0/(2.0*lamda) * np.dot(K,a) - one)
    q = matrix(-b)

    opt = cvxopt.solvers.qp(P=I,q=q,G=G,h=h)
    l = list(opt["x"])
    for i in range(n):
        a[i] = l[i]

    La = (1.0/(4.0 * lamda)) * np.dot(np.transpose(a),np.dot(K,a)) - np.dot(a, one)

    w = (1.0/(2.0 * lamda)) * sum([a[i] * y[i] * x[i] for i in range(n)])
    hin_loss = sum([max((0.0,1.0-y[i]*np.dot(w,x[i]))) for i in range(n)])
    reg = lamda * np.dot(w,w)

    t += 1
    print(t)

    primal.append(hin_loss + reg)
    dual.append(-La)
    gap.append(hin_loss + reg + La)

print(-La)
#print(hin_loss)
#print(reg)
print(hin_loss + reg)
print(hin_loss + reg + La)

plt.plot(range(0,num_iter), primal, linewidth=0.5, markersize=0.5, label='primal')
plt.plot(range(0,num_iter), dual, linewidth=0.5, markersize=0.5, label='dual')
plt.plot(range(0,num_iter), gap, 'go-', linewidth=0.5, markersize=0.5, label='gap')
plt.legend()
plt.xlabel('iter')
plt.ylabel('duality_gap')
plt.show()
