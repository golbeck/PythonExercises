import numpy as np
import matplotlib.pyplot as plt

def main():
    points = np.arange(-5,5,0.1)
    xs,ys=np.meshgrid(points,points)
    print xs
    z=np.sqrt(xs**2+ys**2)
    print z
    plt.imshow(z,cmap=plt.cm.gray);plt.colorbar()

if __name__ == '__main__':
    main()
