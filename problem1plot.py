import numpy as np
import matplotlib.pyplot as plt


num_iter = 300
n = 200
d = 4
lamda = 0.01

#prepare dataset IV
np.random.seed(777)
x_b = 3 * (np.random.rand(n,d) - 0.5)
for i in range(n):
    x_b[i,d-1] = 1
w_b = np.random.randn(d)
y_b = 2 * ((2 * x_b[:,0] - 1 * x_b[:,1] + 0.5 + 0.5 * np.random.randn(n)) > 0) - 1


x_n = x_b
w_n = w_b
y_n = y_b

I = np.identity(d)

#max Lipsitz constant
x_b_T = np.transpose(x_b)
X = np.dot(x_b_T,x_b)
A = X + 2 * lamda * I
eig_val,eig_vec = np.linalg.eig(A)
max_eig = max(eig_val)

#batch steepest gradient method
alpha_b = 1.0 / (4.0*n) * max_eig
loss_hist_batch = []
J_hist_batch = []
wb_hist_batch = []


for t in range(num_iter):

    loss_b = 1.0 / n * sum([np.log(1.0 + np.exp((-y_b[i]) * np.dot(w_b,x_b[i]))) for i in range(n)])
    J_b = loss_b + lamda * np.dot(w_b,w_b)
    loss_hist_batch.append(loss_b)
    J_hist_batch.append(J_b)
    wb_hist_batch.append(w_b)

    direction = 0
    for i in range(n):
        posterior = 1 / (1 + np.exp(-y_b[i] * np.dot(w_b,x_b[i])))
        direction += (1.0-posterior) * (-y_b[i]) * x_b[i]

    direction = direction / n + 2.0 * lamda * w_b
    w_b = w_b - alpha_b * direction

print(w_b)
print(min(loss_hist_batch))
print(J_hist_batch[num_iter-1])
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

    loss_n = 1.0 / n * sum([np.log(1.0 + np.exp((-y_n[i]) * np.dot(w_n,x_n[i]))) for i in range(n)])
    J_n = loss_n + lamda * np.dot(w_n,w_n)
    loss_hist_newton.append(loss_n)
    J_hist_newton.append(J_n)
    wn_hist_newton.append(w_n)

    grad = 0
    hess = 0
    for i in range(n):
        posterior = 1 / (1 + np.exp(-y_n[i] * np.dot(w_n,x_n[i])))
        grad += (1-posterior) * (-y_n[i]) * x_n[i]
        hess += posterior * (1 - posterior) * np.outer(x_n[i],x_n[i])

    grad = grad / n + 2 * lamda * w_n
    hess = hess / n + 2 * lamda * I
    d = np.linalg.solve(hess, -grad)

    w_n = w_n + alpha_n / np.sqrt(t+1) * d


print(w_n)
print(min(loss_hist_newton))
print(J_hist_newton[num_iter-1])
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
