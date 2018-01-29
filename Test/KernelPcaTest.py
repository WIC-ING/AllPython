from scipy.spatial.distance import  pdist, squareform
from scipy import exp
from scipy.linalg import eigh

import numpy as np

import matplotlib.pyplot as plt

def rbf_kernel_pca(X, gamma, n_components):
    """
    RBF kernel PCA implementation.

    :param X: shape=[n_samples, n_features]
    :param gamma:float. Tuning parameter of the RBF kernel

    :param n_components: int. Number of pricipal components to return

    :return: kernel Function Value
    """

    #Calculate pariwise squared Euclidean distances
    sq_dists = pdist(X, 'sqeuclidean')

    #Convert pairwise distances into a square matrix
    mat_sq_dists = squareform(sq_dists)

    #Compute the symmetric kernel matrix.
    K = exp(-gamma * mat_sq_dists)

    #Center the kernel matrix
    N = K.shape[0]
    one_n = np.ones((N,N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    #Obtaining eigenparis from the K'
    eigvals, eigvecs = eigh(K)
    alphas = np.column_stack((eigvecs[:, -i]
                              for i in range(1, n_components + 1)))
    # Collect	the	corresponding	eigenvalues
    lambdas = [eigvals[-i] for i in range(1, n_components + 1)]

    # #Collect the top k eigenvectors(projected samples)
    # X_pc = np.column_stack((eigvecs[:, -i] for i in range(1, n_components+1)))

    return alphas, lambdas
    # return X_pc

# #------------------------------------------------
# #make_circles产生两组圆环形数据集数据集
# #------------------------------------------------
# from sklearn.datasets import make_circles
#
# X, y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)
#
# #显示数据集
# # plt.scatter(X[y==0, 0], X[y==0, 1],
# #             color='red', marker='^', alpha=0.5)
# # plt.scatter(X[y==1, 0], X[y==1, 1],
# #             color='blue', marker='o', alpha=0.5)
# # plt.show()


from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=100, random_state=123)

scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
X_skernpca = scikit_kpca.fit_transform(X)

plt.scatter(X_skernpca[y==0, 0], X_skernpca[y==0, 1],
            color='red', marker='^', alpha=0.5)
plt.scatter(X_skernpca[y==1, 0], X_skernpca[y==1, 1],
            color='blue', marker='o', alpha=0.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()


#------------------------------------------------
#将对一个新样本映射到核空间里
#------------------------------------------------
# from sklearn.datasets import make_moons
#
# X, y = make_moons(n_samples=100, random_state=123)
#
# alphas, lambdas = rbf_kernel_pca(X, gamma=15, n_components=1)
#
# x_new = X[25]
#
# x_proj = alphas[25]
#
# def project_x(x_new, X, gamma, alphas, lambdas):
#     pair_dist = np.array([np.sum((x_new-row)**2) for row in X])
#     k = np.exp(-gamma * pair_dist)
#     return k.dot(alphas/lambdas)
#
# x_reproj = project_x(x_new, X, gamma=15, alphas=alphas, lambdas=lambdas)
#
# plt.scatter(alphas[y==0, 0], np.zeros((50)),
#             color='red', marker='^', alpha=0.5)
#
# plt.scatter(alphas[y==1, 0], np.zeros((50)),
#             color='blue', marker='o', alpha=0.5)
# plt.scatter(x_proj, 0, color='black',
#             label='original projection of point X[25]',
#             marker='^', s=100)
# plt.scatter(x_proj, 0, color='green',
#             label='remapped point X[25]',
#             marker='x', s=500)
# plt.legend(scatterpoints=1)
# plt.show()





#------------------------------------------------
# 核PCA区分圆环形数据集
#------------------------------------------------
# X_kpca = rbf_kernel_pca(X, gamma=15, n_components=2)

#显示标准PCA降维后的结果
# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7,3))
# ax[0].scatter(X_kpca[y==0, 0], X_kpca[y==0, 1],
#               color='red', marker='^', alpha=0.5)
# ax[0].scatter(X_kpca[y==1, 0], X_kpca[y==1, 1],
#               color='blue', marker='o', alpha=0.5)
#
# ax[1].scatter(X_kpca[y==0, 0], np.zeros((500,1))+0.02,
#               color='red', marker='^', alpha=0.5)
# ax[1].scatter(X_kpca[y==1, 0], np.zeros((500,1))-0.02,
#               color='blue', marker='o', alpha=0.5)
#
# ax[0].set_xlabel('PC1')
# ax[0].set_ylabel('PC2')
# ax[1].set_ylim([-1, 1])
# ax[1].set_yticks([])
# ax[1].set_xlabel('PC1')
# plt.show()


# #------------------------------------------------
# #用标准PCA区分圆环形数据集
# #------------------------------------------------
# from sklearn.decomposition import PCA
#
# scikit_pca = PCA(n_components=2)
#
# X_spca = scikit_pca.fit_transform(X)
#
# # #显示标准PCA降维后的结果
# # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7,3))
# # ax[0].scatter(X_spca[y==0, 0], X_spca[y==0, 1],
# #               color='red', marker='^', alpha=0.5)
# # ax[0].scatter(X_spca[y==1, 0], X_spca[y==1, 1],
# #               color='blue', marker='o', alpha=0.5)
# #
# # ax[1].scatter(X_spca[y==0, 0], np.zeros((500,1))+0.02,
# #               color='red', marker='^', alpha=0.5)
# # ax[1].scatter(X_spca[y==1, 0], np.zeros((500,1))-0.02,
# #               color='blue', marker='o', alpha=0.5)
# #
# # ax[0].set_xlabel('PC1')
# # ax[0].set_ylabel('PC2')
# # ax[1].set_ylim([-1, 1])
# # ax[1].set_yticks([])
# # ax[1].set_xlabel('PC1')
# # plt.show()



# #------------------------------------------------
# #用make_moons函数产生100个样本的月牙形线性不可分样本集
# #------------------------------------------------
# from sklearn.datasets import make_moons
#
# X, y = make_moons(n_samples=100, random_state=123)
#
# # plt.scatter(X[y==0, 0], X[y==0, 1],
# #             color='red', marker='^', alpha=0.5)
# #
# # plt.scatter(X[y==1, 0], X[y==1, 1],
# #             color='blue', marker='o', alpha=0.5)
# # plt.show()
# #
#
#
# #------------------------------------------------
# #用Sklearn中的核PCA来对样本数据进行降维
# #------------------------------------------------
# from matplotlib.ticker import FormatStrFormatter
#
# X_kpca = rbf_kernel_pca(X, gamma=15, n_components=2)
#
#
# flg, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
#
# ax1.scatter(X_kpca[y==0, 0], X_kpca[y==0, 1],
#               color='red', marker='^', alpha=0.5)
#
# ax1.scatter(X_kpca[y==1, 0], X_kpca[y==1, 1],
#               color='blue', marker='o', alpha=0.5)
#
# ax2.scatter(X_kpca[y==0, 0], np.zeros((50,1))+0.02,
#               color='red', marker='^', alpha=0.05)
#
# ax2.scatter(X_kpca[y==1, 0], np.zeros((50,1))-0.02,
#               color='blue', marker='o', alpha=0.5)
#
# ax1.set_xlabel('PC1')
# ax1.set_ylabel('PC2')
# ax2.set_ylim([-1, 1])
# ax2.set_yticks([])
# ax2.set_xlabel('PC1')
# ax1.xaxis.set_major_formatter(FormatStrFormatter('%0.f'))
# ax2.xaxis.set_major_formatter(FormatStrFormatter('%0.f'))
# plt.show()
