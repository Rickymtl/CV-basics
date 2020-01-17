import numpy as np
import math
import cv2

def isSeparableFilter(filter):
    u, s, vt = np.linalg.svd(filter)
    z_counter = 0
    for ent in s:
        if round(ent, 3) == 0:
            z_counter = z_counter + 1
    if z_counter != len(s) - 1:
        return 1
    sigma = math.sqrt(s[0])
    ver =  sigma* u[:,0]
    hor =  sigma * vt[0,:]
    print("Vertical filter:")
    print(ver)
    print('Horizontal filter:')
    print(hor)

    return 0


if __name__ == '__main__':
    # gaussian
    k1 = np.array([[0.00730688,  0.03274718,  0.05399097,  0.03274718,  0.00730688],
                   [0.03274718,  0.14676266,  0.24197072,  0.14676266,  0.03274718],
                   [0.05399097,  0.24197072,  0.39894228,  0.24197072,  0.05399097],
                   [0.03274718,  0.14676266,  0.24197072,  0.14676266,  0.03274718],
                   [0.00730688,  0.03274718,  0.05399097,  0.03274718,  0.00730688]])
    print(k1)
    print(isSeparableFilter(k1))

    #
    k2 = np.array([[9, 6, 3, math.sqrt(3)],[3, 2, 1, 0.3333*math.sqrt(3)],[0,0,0,0],[18, 12, 6, 2*math.sqrt(3)]])
    print(k2)
    print(isSeparableFilter(k2))

    k3 = np.array([[1,2,3],[4,5,6],[7,8,9]])
    print(k3)
    print(isSeparableFilter(k3))
