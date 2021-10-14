import numpy as np
from PIL import Image
from Function import restore

if __name__ == "__main__":
    A = Image.open("./scenery.jpg", 'r')
    output_path = r'../result'
    a = np.array(A) # 将输入转化为三维矩阵格式
    print('type(a) = ', type(a))
    print('原始图片大小：', a.shape)

    # 奇异值分解
    # 图片由RGB三原色组成，所以有三个矩阵
    u_r, sigma_r, v_r = np.linalg.svd(a[:, :, 0])
    u_g, sigma_g, v_g = np.linalg.svd(a[:, :, 1])
    u_b, sigma_b, v_b = np.linalg.svd(a[:, :, 2])

    #仅使用前1个，2个，...，1000个奇异值的结果
    K = 1000
    for k in range(1, K+1, 20):
        R = restore(sigma_r, u_r, v_r, k)
        G = restore(sigma_g, u_g, v_g, k)
        B = restore(sigma_b, u_b, v_b, k)
        I = np.stack((R, G, B), axis=2)  # 将矩阵叠合在一起，生成图像
        Image.fromarray(I).save('%s\\svd_%d.jpg' % (output_path, k))

    # k = 500
    # R = restore(sigma_r, u_r, v_r, k)
    # G = restore(sigma_g, u_g, v_g, k)
    # B = restore(sigma_b, u_b, v_b, k)
    # I = np.stack((R, G, B), axis=2)  # 将矩阵叠合在一起，生成图像
    # Image.fromarray(I).save('%s\\svd_%d.jpg' % (output_path, k))
