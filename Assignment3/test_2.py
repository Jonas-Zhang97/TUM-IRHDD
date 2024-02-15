import numpy as np
def gram_pca(K,k):
    # Insert Code for Subtask 1.2 here
    N = K.shape[0]
    one = np.ones((N, N))
    H = np.eye(N) - one/N
    K_tilde = H @ K @ H
    # Compute the eigenvectors and eigenvalues of K_tilde
    eigenvalues, eigenvectors = np.linalg.eigh(K_tilde)
    # Sort the eigenvectors and eigenvalues in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    # Compute the first k principal components
    S = np.diag(np.sqrt(eigenvalues[:k])) @ eigenvectors[:, :k].T
    return S

# Insert Code for Subtask 1.1 here
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import imageio
import os
directory = '/Users/cheng/Downloads/mnist'
num_samples = []
for i in range(10):
    # num_samples.append(100)
    # Uncomment the next line and comment the previous line to load all images, but might lead to out of RAM
    num_samples.append(len(os.listdir(directory + '/' + 'd' + str(i))))
N = sum(num_samples)
X = np.zeros((28*28, sum(num_samples)))
Y = np.zeros(N)
for i in range(10):
    folder = directory + '/' + 'd' + str(i)
    cnt = 0
    for j, file in enumerate(os.listdir(folder)):
        img = imageio.imread(folder + '/' + file).reshape(28*28)
        X[:, sum(num_samples[:i]) + j] = img
        Y[sum(num_samples[:i]) + j] = i
        cnt += 1
        if cnt == num_samples[i]:
            break

# Insert Code for Subtask 1.3 here
k = 2
K = np.dot(X.T, X)
S_gram = gram_pca(K, k)

# Draw the scatter plot with different colors for different digits
np.random.seed(0)
perm = np.random.permutation(N)
S_gram = S_gram[:, perm]
Y_shuffled = Y[perm]
colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'orange', 'purple', 'pink']
C = [colors[int(label)] for label in Y_shuffled]
# for i in range(10):
#     C[Y==i] = np.array([int(c) for c in colors[i]], dtype=float)/255
plt.scatter(S_gram[0, :], S_gram[1, :], c=C)
legend_patch = [mpatches.Patch(color=colors[i], label=str(i)) for i in range(0, 10)]
plt.legend(handles=legend_patch)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('2D PCA of MNIST')
plt.show()
# Save the scatter plot
plt.savefig('2D_PCA_of_MNIST.png')