import numpy as np
import matplotlib.pyplot as plt

TD = np.load('TrainDigits.npy')
TL = np.load('TrainLabels.npy')

# Sorting the digits
TL_flat = TL.flatten()

A = []
for  i in range(10):
    index = (TL_flat == i)  # boolean array with index of where digit is True
    digits_i = TD[:, index]  # matching index with columns
    A.append(digits_i[:, : 400])

# Computing SVD
Sigma = []
U = []
VT = []

for i in range(10):
    Ai = A[i]
    ATA = Ai.T @ Ai

    eigvals, V = np.linalg.eigh(ATA)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    V = V[:, idx]
    eigvals = np.maximum(eigvals, 0) # remove negative values
    
    singular_values = np.sqrt(eigvals)
    r = np.sum(singular_values > 1e-8)  # remove singular values close to zero
    singular_values = singular_values[:r]
    V = V[:, :r]

    Ui = np.zeros((Ai.shape[0], r))
    for j in range(r):
        Ui[:, j] = (Ai @ V[:, j]) / singular_values[j]

    # orthonormalization
    Ui, _ = np.linalg.qr(Ui)

    U.append(Ui)
    Sigma.append(singular_values)
    VT.append(V.T)
# 
# Plotting singular values
digits = [Sigma[3], Sigma[8]]

for r in range(2):
    sv = digits[r]
    fig, ax = plt.subplots()
    ax.plot(sv)
    if r==0:
        ax.set_title(f'Singular values for digit 3')
    elif r==1:
        ax.set_title(f'Singular values for digit 8')

    plt.show()

# Plotting singular images
digits = [U[3], U[8]]

for r in range(2):
    for c in range(3):
        fig, ax = plt.subplots()
        img = np.reshape(digits[r][:, c], (28, 28)).T
        ax.imshow(np.abs(img), cmap='gray')
        ax.axis('off')
        if r==0:
            ax.set_title(f'Singular image nr. {c+1} for digit 3')
        elif r==1:
            ax.set_title(f'Singular image nr. {c+1} for digit 8')

        plt.show()
