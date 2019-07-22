import numpy as np
import random
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

K = np.zeros((n,n), dtype = float)

x = np.random.randn(n,d) + 0.05
y = np.zeros(n,dtype = int)
for i in range(n):
    y[i] = 2 * (np.dot(omega,x[i]) + noise > 0) - 1
for i in range(n):
    for j in range(n):
        K[i,j] = y[i] * y[j] * np.dot(x[i],x[j])


zero = np.zeros(n, dtype = int)
one = np.ones(n, dtype = int)

#initial state for projected gradient
a = np.random.rand(n)
#initial state for dual coordinate descent
a_dc = np.zeros(n, dtype = float)
w_dc = np.zeros(d, dtype = float)

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
print(hin_loss)
print(reg)
print(hin_loss + reg + La)

plt.plot(range(0,num_iter), primal, linewidth=0.5, markersize=0.5, label='primal')
plt.plot(range(0,num_iter), dual, linewidth=0.5, markersize=0.5, label='dual')
plt.plot(range(0,num_iter), gap, 'go-', linewidth=0.5, markersize=0.5, label='gap')
plt.legend()
plt.xlabel('iter')
plt.ylabel('duality_gap')
plt.show()

#dual coordinate descent
gap_dc = []
primal_dc = []
dual_dc = []
b = list(range(n))
for t in range(num_iter):
    random.shuffle(b)
    for i in range(n):
        temp = a_dc[b[i]]
        G = y[b[i]] * np.dot(w_dc,x[b[i]]) - 1
        if(a_dc[b[i]] == 0):
            PG = min((G, 0))
        elif(a_dc[b[i]] == 1):
            PG = max((G, 0))
        else:
            PG = G
        if(PG != 0):
            a_dc[b[i]] = min((max((0, a_dc[b[i]] - G / (1 / (2 * lamda) * K[b[i],b[i]]))),1))
            w_dc = w_dc + (a_dc[b[i]] - temp) * y[b[i]] * x[b[i]]

    La_dc = (1.0/(4.0 * lamda)) * np.dot(np.transpose(a_dc),np.dot(K,a_dc)) - np.dot(a_dc, one)
    hin_loss_dc = sum([max((0.0,1.0-y[j]*np.dot(w_dc,x[j]))) for j in range(n)])
    reg_dc = lamda * np.dot(w_dc,w_dc)
    primal_dc.append(hin_loss_dc + reg_dc)
    dual_dc.append(-La)
    gap_dc.append(hin_loss_dc + reg_dc + La_dc)

print(-La_dc)
print(hin_loss_dc)
print(reg_dc)
print(hin_loss_dc + reg_dc + La_dc)


plt.plot(range(num_iter), primal_dc, linewidth=0.5, markersize=0.5, label='primal_dc')
plt.plot(range(num_iter), dual_dc, linewidth=0.5, markersize=0.5, label='dual_dc')
plt.plot(range(num_iter), gap_dc, 'go-', linewidth=0.5, markersize=0.5, label='gap_dc')
plt.legend()
plt.xlabel('iter')
plt.ylabel('dual_coordinate')
plt.show()


#compare
show_iter = 100

plt.plot(gap[:show_iter], linewidth=0.5, markersize=1, label='projected_gradient')
plt.plot(gap_dc[:show_iter], linewidth=0.5, markersize=1, label='dual_coordinate')
plt.legend()
plt.xlabel('iter')
plt.ylabel('convergence of duality_gap')
plt.show()
