import numpy as np
def restore(sigma, u, v, K):  # 奇异值、左奇异值、右奇异值
    m = len(u)
    n = len(v[0])
    a = np.zeros((m, n))# 生成m行n列的零矩阵
    for k in range(K):
        uk = u[:, k].reshape(m, 1)# 取每行第k个元素
        vk = v[k].reshape(1, n)
        a += sigma[k] * np.dot(uk, vk)  # 前k个奇异值的加和
    a = a.clip(0, 255)
    return np.rint(a).astype('uint8')