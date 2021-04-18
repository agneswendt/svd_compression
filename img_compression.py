from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys

def main(f_name):
    img = Image.open(f_name).convert('RGB')
    red, green, blue = img.split()

    svd_r = np.linalg.svd(np.array(red))
    svd_g = np.linalg.svd(np.array(green))
    svd_b = np.linalg.svd(np.array(blue))

    full_rank = red.size[1]

    for i in [full_rank, 500, 100, 20, 5]:
        r, g, b = compress(svd_r, i), compress(svd_g, i), compress(svd_b, i)
        reconst = np.dstack((r,g,b))
        plt.imshow(reconst)
        title = f"rank {np.linalg.matrix_rank(r)}"
        plt.title(title)
        plt.show()


def compress(svd, rank):
    U, sigma, V = svd
    res = np.matrix(U[:, :rank]) * np.diag(sigma[:rank]) * np.matrix(V[:rank, :])
    return np.true_divide(np.array(res), 255)


if __name__=="__main__":
    main(sys.argv[1])
