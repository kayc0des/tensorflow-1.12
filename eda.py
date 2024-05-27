#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

# load the data
lib = np.load('data/MNIST.npz')
X_test_3D = lib['X_test']
Y_test = lib['Y_test']

print('shape of X_test_3D is {}'.format(X_test_3D.shape))
print('shape of Y_test is {}'.format(Y_test.shape))

fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_test_3D[i])
    plt.title(Y_test[i])
    plt.axis('off')
plt.tight_layout()
plt.show()
