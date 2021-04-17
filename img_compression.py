from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys

def main(f_name):
    img = Image.open(f_name).convert('RGB')
    red, green, blue = img.split()

    red_img = np.array(red)
    red_img.shape = (red.size[1], red.size[0])

    green_img = np.array(green)
    green_img.shape = (green.size[1], green.size[0])

    blue_img = np.array(blue)
    blue_img.shape = (blue.size[1], blue.size[0])

    print(red_img, blue_img, green_img)

    plt.figure(figsize=(9,6))
    rgb = to_matrix(red_img, green_img, blue_img, red.size[0], red.size[1])

    print()

    plt.imshow(rgb)
    plt.title(f"original, matrix rank {np.linalg.matrix_rank(red_img)}")
    plt.show()

    svd_r = np.linalg.svd(red_img)
    svd_g = np.linalg.svd(green_img)
    svd_b = np.linalg.svd(blue_img)

    for i in [5, 20, 100, 500]:
        reconst = to_matrix(compress(svd_r, i), compress(svd_g, i),
                            compress(svd_b, i), red.size[0], red.size[1])
        plt.imshow(reconst)
        title = f"rank {np.linalg.matrix_rank(compress(svd_r, i))}"
        plt.title(title)
        plt.show()


def to_matrix(r, g, b, w, h):
    res = []
    for y in range(h):
        row = []
        for x in range(w):
            if r[y, x]/255 > 1: print(r[y,x])
            row.append((r[y, x]/255, g[y, x]/255, b[y, x]/255))
        res.append(row)
    return np.array(res)


def compress(svd, rank):
    U, sigma, V = svd
    return np.matrix(U[:, :rank]) * np.diag(sigma[:rank]) * np.matrix(V[:rank, :])



if __name__=="__main__":
    main(sys.argv[1])
