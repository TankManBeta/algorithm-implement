from sklearn import datasets,metrics
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
import numpy as np
import matplotlib.pyplot as plt

# 用sklearn生成数据集
def generate_data(n_point=400):
    xx, color = datasets.samples_generator.make_swiss_roll(n_point, random_state=9)
    return xx, color

# 计算k近邻矩阵，并且使用floyd算法算出任意两点间的最短距离
def floyd(D, n_neighbors=10):
    Max = np.max(D)*1000
    n1,n2 = D.shape
    k = n_neighbors
    D1 = np.ones((n1,n1))*Max
    D_arg = np.argsort(D, axis=1)
    for i in range(n1):
        D1[i,D_arg[i,0:k+1]] = D[i,D_arg[i,0:k+1]]
    for h in range(n1):
        for i in range(n1):
            for j in range(n1):
                if D1[i,h]+D1[h,j]<D1[i,j]:
                    D1[i,j] = D1[i,h]+D1[h,j]
    return D1

# 计算欧式距离
def calculate_distance_matrix(x, y):
    d = metrics.pairwise_distances(x, y)
    return d

# 用于计算MDS算法中的B，即计算内积矩阵
def cal_inner_product_matrix(D):
    (n1, n2) = D.shape
    # 求距离矩阵的平方
    DD = np.square(D)
    Di = np.sum(DD, axis=1) / n1
    Dj = np.sum(DD, axis=0) / n1
    Dij = np.sum(DD) / (n1 ** 2)
    B = np.zeros((n1, n1))
    for i in range(n1):
        for j in range(n2):
            B[i, j] = (Dij + DD[i, j] - Di[i] - Dj[j]) / (-2)
    return B

# 合并成为ISOMAP算法
def Isomap(data, n_dimension=2, n_neighbors=10):
    # 计算距离矩阵
    D = calculate_distance_matrix(data, data)
    # 计算k最近邻矩阵，并用floyd算法求 最短距离矩阵
    D_floyd = floyd(D)
    # 计算内积矩阵
    B = cal_inner_product_matrix(D_floyd)
    # 以下步骤对内积矩阵进行特征值分解
    Be,Bv = np.linalg.eigh(B)
    # 将特征值降序排列
    Be_sort = np.argsort(-Be)
    Be = Be[Be_sort]
    Bv = Bv[:,Be_sort]
    Bez = np.diag(Be[0:n_dimension])
    Bvz = Bv[:, 0:n_dimension]
    Z = np.dot(np.sqrt(Bez), Bvz.T).T
    # print(Z.shape)
    return Z


if __name__=='__main__':
    data,color=generate_data(400)
    Z_Isomap=Isomap(data, n_dimension=2)
    fig=plt.figure(figsize=(6,4))
    gs = fig.add_gridspec(1,2)
    ax1 = fig.add_subplot(gs[0,0],projection='3d')
    # 第1幅子图表示原始样本分布
    ax1.scatter(data[:, 0], data[:, 1], data[:, 2], c=color, cmap=plt.cm.Spectral)
    ax2 = fig.add_subplot(gs[0, 1])
    # 第2副子图表示降维后样本分布
    ax2.scatter(Z_Isomap[:, 0], Z_Isomap[:, 1], c=color, cmap=plt.cm.Spectral)
    ax2.xaxis.set_major_formatter(NullFormatter())
    ax2.yaxis.set_major_formatter(NullFormatter())
    plt.show()