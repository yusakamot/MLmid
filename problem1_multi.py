import numpy as np
import matplotlib.pyplot as plt


num_iter =300
n = 200
d = 3
lamda = 0.01

#prepare dataset IV
np.random.seed(777)

#for steepest
x_b = 3 * (np.random.rand(n,d) - 0.5)
W_b = np.array([[ 2, -1, 0.5],
             [-3,  2,   1],
             [ 1,  2,  3]])
y_b = np.argmax(np.dot(np.hstack([x_b[:,:d-1], np.ones((n, 1))]), W_b.T)
                        + 0.5 * np.random.randn(n, d), axis=1)

#for newton
x_n = x_b
W_n = W_b
y_n = y_b

I = np.identity(d)

#batch simple steepest gradient method
loss_hist_batch = []
J_hist_batch = []
wb_hist_batch = []

for t in range(num_iter):
    alpha_b = 1.0 / np.sqrt(t+1)

    loss_b = 1 / n * sum([-np.dot(W_b[y_b[i]],x_b[i]) + np.log(sum([np.exp(np.dot(W_b[j],x_b[i])) for j in range(3)])) for i in range(n)])
    J_b = loss_b + lamda * sum([np.dot(W_b[k],W_b[k]) for k in range(3)])
    loss_hist_batch.append(loss_b)
    J_hist_batch.append(J_b)
    wb_hist_batch.append(W_b)

    direction = [0,0,0]
    for j in range(3):
        for i in range(n):
            direction[j] += (np.exp(np.dot(W_b[j],x_b[i])) * x_b[i]) / sum([np.exp(np.dot(W_b[k],x_b[i])) for k in range(3)])
            if(y_b[i] == j):
                direction[j] += -x_b[i]
        direction[j] = 1 / n * direction[j]
        direction[j] += 2.0 * lamda * W_b[j]
    direction = np.array(direction)
    W_b = W_b - alpha_b * direction


print(W_b)
print(min(loss_hist_batch))
print(min(J_hist_batch))

plt.plot(range(num_iter), J_hist_batch, 'bo-', linewidth=0.5, markersize=0.5, label='steepest')
plt.legend()
plt.xlabel('iter')
plt.ylabel('loss')
plt.show()


#newton based method
alpha_n = 1.0
loss_hist_newton = []
J_hist_newton = []
wn_hist_newton = []

for t in range(num_iter):

    loss_n = 1 / n * sum([-np.dot(W_n[y_n[i]],x_n[i]) + np.log(sum([np.exp(np.dot(W_n[j],x_n[i])) for j in range(3)])) for i in range(n)])
    J_n = loss_n + lamda * sum([np.dot(W_n[k],W_n[k]) for k in range(3)])
    loss_hist_newton.append(loss_n)
    J_hist_newton.append(J_n)
    wn_hist_newton.append(W_n)

    grad = [0,0,0]
    for j in range(3):
        for i in range(n):
            grad[j] += (np.exp(np.dot(W_n[j],x_n[i])) * x_n[i]) / sum([np.exp(np.dot(W_n[k],x_n[i])) for k in range(3)])
            if(y_n[i] == j):
                grad[j] += -x_n[i]
        grad[j] = 1 / n * grad[j]
        grad[j] += 2.0 * lamda * W_n[j]
    grad = np.array(grad)


    hess = [0,0,0]
    sumk = (sum([np.exp(np.dot(W_n[k],x_n[i])) for k in range(3)]))**2
    for j in range(3):
        for i in range(n):
            hess[j] += (np.exp(np.dot(W_n[j],x_n[i])) * sum([np.exp(np.dot(W_n[k],x_n[i])) for k in range(3) if k != j])) / sumk * np.outer(x_n[i],x_n[i])
        hess[j] = 1 / n * hess[j]
        hess[j] += 2 * lamda * I
    hess = np.array(hess)


    d = [0,0,0]
    for j in range(3):
        d[j] = np.linalg.solve(hess[j], -grad[j])
    d = np.array(d)


    W_n = W_n + alpha_n * d


print(W_n)
print(min(loss_hist_newton))
print(min(J_hist_newton))


plt.plot(range(num_iter), J_hist_newton, 'ro-', linewidth=0.5, markersize=0.5, label='newton')
plt.legend()
plt.xlabel('iter')
plt.ylabel('loss')
plt.show()

show_iter = 50
min_J = min(min(J_hist_batch), min(J_hist_newton))

plt.plot(np.abs(J_hist_batch[:show_iter] - min_J), 'bo-', linewidth=0.5, markersize=1, label='steepest')
plt.plot(np.abs(J_hist_newton[:show_iter] - min_J), 'ro-', linewidth=0.5, markersize=1, label='newton')
plt.legend()
plt.yscale('log')
plt.xlabel('iter')
plt.ylabel('diff from the gold weight')
plt.show()
