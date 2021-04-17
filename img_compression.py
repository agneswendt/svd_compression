from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys

def main(f_name):
    img = Image.open(f_name)
    img_gray = img.convert("LA")
    imgmat = np.array(list(img_gray.getdata(band=0)), float)
    imgmat.shape = (img_gray.size[1], img_gray.size[0])
    imgmat = np.matrix(imgmat)
    plt.figure(figsize=(9,6))
    plt.imshow(imgmat, cmap='gray');
    plt.title(f"original, matrix rank {np.linalg.matrix_rank(imgmat)}")
    plt.show()
    U, sigma, V = np.linalg.svd(imgmat)
    for i in [55, 20, 100, 500]:
        reconstimg = np.matrix(U[:, :i]) * np.diag(sigma[:i]) * np.matrix(V[:i, :])
        plt.imshow(reconstimg, cmap='gray')
        title = f"rank {np.linalg.matrix_rank(reconstimg)}"
        plt.title(title)
        plt.show()


if __name__=="__main__":
    main(sys.argv[1])
