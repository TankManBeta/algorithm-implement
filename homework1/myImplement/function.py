import numpy as np
def mysvd(a):
    # a = np.array([[16,8],
    #               [4,9],
    #               [7,8]])
    a_ts = a.transpose()
    c = a.dot(a_ts)
    sigma,sigma_vector = np.linalg.eigh(c)
    sigma_sort_index = np.argsort(sigma)[::-1]
    sigma = np.sort(sigma)[::-1]
    for i in range(len(sigma)):
        if sigma[i] < 1e-20:
            sigma[i] = 1e-20
    sigma_vector = sigma_vector[:,sigma_sort_index]
    sigma = np.sqrt(sigma)
    sigma_invert = np.linalg.inv(np.diag(sigma))
    sigma_vector_invert = np.linalg.inv(sigma_vector)

    # t1,t2,t3 = np.linalg.svd(a)

    v_invert = (sigma_invert.dot(sigma_vector_invert)).dot(a)
    print(sigma_vector_invert)
    print(sigma)
    print(v_invert)
    return sigma_vector_invert,sigma,v_invert
