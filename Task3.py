import numpy as np
import matplotlib.pyplot as plt
from MiniProject2 import U


TestD = np.load('TestDigits.npy')
TestL = np.load('TestLabels.npy').flatten()

# Computing U_k and U_k*U_k^T
Uk = []
Pk = []

for i in range(10):
    Ui = U[i]
    Uk_i = []
    Pk_i = []

    for k in range(5, 16):
        U_ik = Ui[:, :k]
        P_ik = U_ik @ U_ik.T
        Uk_i.append(U_ik)
        Pk_i.append(P_ik)

    Uk.append(Uk_i)
    Pk.append(Pk_i)

# Residuals
residuals = np.zeros((10, 11, TestD.shape[1]))

for i in range(10):
    for j, k in enumerate(range(5, 16)):
        P = Pk[i][j]
        res = TestD - P @ TestD
        residuals[i, j, :] = np.linalg.norm(res, axis=0)


# Classification
digit_accuracies = np.zeros((10, 11))

for j, k in enumerate(range(5, 16)):
    predictions = np.argmin(residuals[:, j, :], axis=0)

    print(f"\nk = {k}")

    for digit in range(10):
        mask = (TestL == digit)  # correct
        acc = np.mean(predictions[mask] == digit)  # accuracy for this digit
        digit_accuracies[digit, j] = acc
        print(f"Digit {digit}: {acc*100:.2f}%")
        