import numpy as np
import matplotlib.pyplot as plt

def solve_homography(u, v):
    """
    This function should return a 3-by-3 homography matrix,
    u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
    :param u: N-by-2 source pixel location matrices
    :param v: N-by-2 destination pixel location matrices
    :return:
    """

    N = u.shape[0]
    H = None

    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')

    # TODO: 1.forming A
    A = []
    for i in range(N):
        ux = u[i][0]
        uy = u[i][1]
        vx = v[i][0]
        vy = v[i][1]
        A.append([ux,uy,1,0,0,0,-ux*vx,-uy*vx,-vx])
        A.append([ux,0,0,0,uy,1,-ux*vy,-uy*vy,-vy])
    #print(A)
    # TODO: 2.solve H with A
    U,S,V = np.linalg.svd(A)
    last = V.shape[1]-1
    H = V[last]
    H = np.array(H)
    sum_of_H = 0
    #for i in range(len(H)):
    #    sum_of_H+=H[i]
    #print("sum of H")
    #print(sum_of_H)
    H = np.reshape(H,(3,3))
    #print(H)
    return H


def warping(src, dst, H, ymin, ymax, xmin, xmax, direction='b'):
    """
    Perform forward/backward warpping without for loops. i.e.
    for all pixels in src(xmin~xmax, ymin~ymax),  warp to destination
          (xmin=0,ymin=0)  source                       destination
                         |--------|              |------------------------|
                         |        |              |                        |
                         |        |     warp     |                        |
    forward warp         |        |  --------->  |                        |
                         |        |              |                        |
                         |--------|              |------------------------|
                                 (xmax=w,ymax=h)

    for all pixels in dst(xmin~xmax, ymin~ymax),  sample from source
                            source                       destination
                         |--------|              |------------------------|
                         |        |              | (xmin,ymin)            |
                         |        |     warp     |           |--|         |
    backward warp        |        |  <---------  |           |__|         |
                         |        |              |             (xmax,ymax)|
                         |--------|              |------------------------|

    :param src: source image
    :param dst: destination output image
    :param H:
    :param ymin: lower vertical bound of the destination(source, if forward warp) pixel coordinate
    :param ymax: upper vertical bound of the destination(source, if forward warp) pixel coordinate
    :param xmin: lower horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param xmax: upper horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param direction: indicates backward warping or forward warping
    :return: destination output image
    """

    h_src, w_src, ch = src.shape
    h_dst, w_dst, ch = dst.shape
    H_inv = np.linalg.inv(H)
    
    #print(xmin,xmax,ymin,ymax)
    #print("src:")
    #print(h_src,w_src,ch)
    #print("dst:")
    #print(h_dst,w_dst,ch)
    #print(dst)
    # TODO: 1.meshgrid the (x,y) coordinate pairs
    # TODO: 2.reshape the destination pixels as N x 3 homogeneous coordinate
    N = h_dst * w_dst
    dst = np.reshape(dst,(N,3))
    print(dst.shape)
    if direction == 'b':
        x,y = np.arange(w_dst), np.arange(h_dst)
        X,Y = np.meshgrid(x,y)
        X,Y = X.flatten(), Y.flatten()
        l = len(X)
        Z = np.ones((l,))
        V = [X,Y,Z]
        V = np.array(V)
        # TODO: 3.apply H_inv to the destination pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        U = np.matmul(H_inv,V)
        U = U / U[-1,:]
        print(U)
        #print(U_x)
        #print(U_y)
        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of source image)
        del1 = np.argwhere(U_x<0)
        del2 = np.argwhere(U_y<0)
        del3 = np.argwhere(U_x>w_src-1)
        del4 = np.argwhere(U_y>h_src-1)
        exceed = np.unique(np.concatenate((del1,del2,del3,del4),axis = 0))
        U_x, U_y = np.delete(U_x,exceed), np.delete(U_y,exceed)
        # TODO: 5.sample the source image with the masked and reshaped transformed coordinates

        # TODO: 6. assign to destination image with proper masking

        pass

    elif direction == 'f':
        x,y = np.arange(w_src), np.arange(h_src)
        X,Y = np.meshgrid(x,y)
        X,Y = X.flatten(), Y.flatten()
        l = len(X)
        Z = np.ones((l,))
        U = [X,Y,Z]
        U = np.array(U)
        # TODO: 3.apply H to the source pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        V = np.matmul(H,U)
        V = V/V[-1,:]
        V_x = V[:1, :].flatten()
        V_y = V[1:2,:].flatten()
        #print(V_x)
        #print(V_y)
        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of destination image)

        # TODO: 5.filter the valid coordinates using previous obtained mask

        # TODO: 6. assign to destination image using advanced array indicing

        pass

    return dst
