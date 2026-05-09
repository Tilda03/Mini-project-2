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


# Computing the SVD
Sigma = []
U = []
VT = []

for i in range(10):
    Ai = A[i]

    # Singular values
    eigval_singval_i = np.abs(np.linalg.eigvals(Ai.T @ Ai))
    singval_i = np.sqrt(eigval_singval_i)

    # idx = np.argsort(singval_i)[::-1]
    # singval_i = singval_i[idx]

    Sigma.append(np.diag(singval_i))

    # U
    eigval_U_i, eigvec_U_i = np.linalg.eigh(Ai @ Ai.T)
    idx = np.argsort(eigval_U_i)[::-1]

    eigvec_U_i = eigvec_U_i[:, idx]
    U.append(eigvec_U_i)

    # V^T
    eigval_V_i, eigvec_V_i = np.linalg.eigh(Ai.T @ Ai)
    idx = np.argsort(eigval_V_i)[::-1]

    eigvec_V_i = eigvec_V_i[:, idx]
    VT.append(eigvec_V_i.T)


# Plotting singular values
digits = [Sigma[3], Sigma[8]]

for r in range(2):
    sv = np.diag(digits[r])  # extracting the diagonal
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
