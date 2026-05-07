import numpy as np
import matplotlib.pyplot as plt

TD = np.load('TrainDigits.npy')
TL = np.load('TrainLabels.npy')

TL_flat = TL.flatten()

A = []
for  i in range(10):
    index = (TL_flat == i)  # boolean array with index of where digit is True
    digits_i = TD[:, index]  # matching index with columns
    A.append(digits_i[:, : 400])



Sigma = []
U = []

for i in range(10):
    Ai = A[i]

    # Singular values
    eigval_singval_i = np.abs(np.linalg.eigvals(Ai.T @ Ai))
    singval_i = np.sqrt(eigval_singval_i)
    Sigma.append(singval_i)

    # U
    eigval_U_i, eigvec_U_i = np.linalg.eigh(Ai @ Ai.T)
    idx = np.argsort(eigval_U_i)[::-1]

    eigvec_U_i = eigvec_U_i[:, idx]
    U.append(eigvec_U_i)



digits = [Sigma[3], Sigma[8]]
fig, axes = plt.subplots(2, 1)

for r in range(2):
    axes[r].plot(digits[r])

plt.tight_layout()
plt.show()


digits = [U[3], U[8]]

fig, axes = plt.subplots(2, 3)

for r in range(2):
    for c in range(3):
        img = np.reshape(digits[r][:, c], (28, 28)).T
        #axes[r, c].imshow(img, cmap='gray')
        #axes[r, c].imshow(img, cmap='seismic'r, c)
        axes[r, c].imshow(np.abs(img), cmap='gray')
        axes[r, c].axis('off')

plt.tight_layout()
plt.show()


#sum = np.sum(index)



#print(digits)

#d = digits[:,56]

#D = np.reshape(d, (28, 28)).T

#d = TrainDigits[:,0]

#l = TrainLabels[:,0]

#D = np.reshape(d, (28, 28)).T

#print(k)




#plt.imshow(D, cmap ='gray')
#plt.show()