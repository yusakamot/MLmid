lam2 = 2
lam4 = 4
lam6 = 6

import numpy as np
import matplotlib.pyplot as plt

def st_ops(mu, q):
    x_proj = np.zeros(mu.shape)
    for i in range(len(mu)):
        if(mu[i] > q):
            x_proj[i] = mu[i] - q
        elif(np.abs(mu[i]) < q):
            x_proj[i] = 0
        else :
            x_proj[i] = mu[i] + q
    return x_proj

"""
x_1 = np.arange(-1.5, 3, 0.01)
x_2 = np.arange(-1.5, 3, 0.02)

X1, X2 = np.mgrid[-1.5:3:0.01, -1.5:3:0.02]
fValue = np.zeros((len(x_1), len(x_2)))
"""
A = np.array([[  3, 0.5],
              [0.5,   1]])
mu = np.array([[1],
               [2]])

"""
for i in range(len(x_1)):
    for j in range(len(x_2)):
        inr = np.vstack([x_1[i], x_2[j]])
        fValue[i, j] = np.dot(np.dot((inr-mu).T, A), (inr- mu)) + lam2 * (np.abs(x_1[i]) + np.abs(x_2[j]))
"""

w_init2 = np.array([[ 3],
                   [-1]])
w_init4 = np.array([[ 3],
                   [-1]])
w_init6 = np.array([[ 3],
                   [-1]])

eig_val,eig_vec = np.linalg.eig(2 * A)
L = max(eig_val)

w_history2 = []
w_history4 = []
w_history6 = []

Val_history2 = []
Val_history4 = []
Val_history6 = []

wt2 = w_init2
wt4 = w_init4
wt6 = w_init6

wopt2 = np.array([[0.82],[1.09]])
wopt4 = np.array([[0.64], [0.18]])
wopt6 = np.array([[0.33], [0]])
for t in range(1000):
  w_history2.append(np.transpose(wt2))
  w_history4.append(np.transpose(wt4))
  w_history6.append(np.transpose(wt6))

  Val_history2.append(np.sqrt(float(np.dot((wt2-wopt2).T, wt2-wopt2))))
  Val_history4.append(np.sqrt(float(np.dot((wt4-wopt4).T, wt4-wopt4))))
  Val_history6.append(np.sqrt(float(np.dot((wt6-wopt6).T, wt6-wopt6))))

  grad2 = 2 * np.dot(A, wt2-mu)
  grad4 = 2 * np.dot(A, wt4-mu)
  grad6 = 2 * np.dot(A, wt6-mu)

  wth = wt2 - 1/L * grad2 #gradient descent update
  wt2 = st_ops(wth, lam2 * 1 / L)  #1要素ごとのSoft thresholding

  wth = wt4 - 1/L * grad4 #gradient descent update
  wt4 = st_ops(wth, lam4 * 1 / L)  #1要素ごとのSoft thresholding

  wth = wt6 - 1/L * grad6 #gradient descent update
  wt6 = st_ops(wth, lam6 * 1 / L)  #1要素ごとのSoft thresholding

print(wt2)
print(wt4)
print(wt6)

w_history2.append(np.transpose(wt2))
w_history2 = np.vstack(w_history2)

w_history4.append(np.transpose(wt4))
w_history4 = np.vstack(w_history4)

w_history6.append(np.transpose(wt6))
w_history6 = np.vstack(w_history6)

"""
plt.contour(X1,X2,fValue)
plt.plot(w_history[:,0], w_history[:,1], 'ro-', markersize=3, linewidth=0.5)

plt.xlim(-1.5, 3)
plt.ylim(-1.5, 3)
plt.show()
"""
show_iter = 1000
plt.plot(range(show_iter), Val_history2, linewidth=0.5, markersize=1, label='lambda2')
plt.plot(range(show_iter), Val_history4, linewidth=0.5, markersize=1, label='lambda4')
plt.plot(range(show_iter), Val_history6, linewidth=0.5, markersize=1, label='lambda6')
plt.legend()
plt.yscale('log')
plt.xlabel('iter')
plt.ylabel('diff from the gold weight')
plt.show()
