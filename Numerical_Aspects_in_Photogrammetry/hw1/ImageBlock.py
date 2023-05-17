from Camera import Camera
from SingleImage import SingleImage
from MatrixMethods import Compute3DRotationMatrix, Compute3DRotationDerivativeMatrix, ComputeSkewMatrixFromVector
import numpy as np
from numpy import linalg as la
import pandas as pd
import PhotoViewer as pv
from matplotlib import pyplot as plt
# from ImagePair import ImagePair
# import datetime
# import xlsxwriter

class ImageBlock(object):
    def __init__(self, images, tie_points, control_points):
        """
        Initialize the ImageBlock class
        :param images: images of the block
        :param control_points: ground control coordinates
        :param tie_points: tie points samples in camera system
        :param GC_data: ground samples in camera system

        :type images: list of SingleImage
        :type control_points: data frame with columns=['x', 'y','z' 'name', 'image_id', 'X', 'Y', 'Z']
        :type tie_points: data frame with columns=['x', 'y', 'name', 'image_id', 'X', 'Y', 'Z']
        :type GC_data: float np.array tx4:[image number, point number,x,y]
        """
        self.__images = images
        self.__control_points = control_points
        self.__tie_points = tie_points


    # ---------------------- Properties ----------------------
    
    # create property for tie points
    
    @property
    def tie_points(self):
        return self.__tie_points
    
    @tie_points.setter
    def tie_points(self, val):
        self.__tie_points = val
       
    @property
    def scale(self):
        alt = self.images[0].exteriorOrientationParameters[2]
        f = self.images[0].camera.focal_length
        return f/alt
    
    @property
    def images(self):
        """
        images

        :return: images

        :rtype: list of SingleImage

        """
        return self.__images
    @property
    def control_points(self):
        return self.__control_points
    
    @control_points.setter
    def control_points(self, val):
        self.__control_points = val
              
    
    @property
    def T_coordinates(self):
        """
        Tie points coordinates

        :return: Tie points coordinates

        :rtype: np.array nx4: [point number,X,Y,Z]

        """
        return self.__T_coordinates

    @T_coordinates.setter
    def T_coordinates(self, val):
        """
        Tie points coordinates

        :param val: Tie points coordinates

        :type: np.array nx4: [point number,X,Y,Z]

        """

        self.__T_coordinates = val
        
    @property
    def block_boundaries(self):
        # get images boundaries and find the min and max
        for image in self.images:
            image_boundaries = image.image_ground_bounds
            if 'xmin' in locals():
                xmin = min(xmin, image_boundaries['xmin'])
                xmax = max(xmax, image_boundaries['xmax'])
                ymin = min(ymin, image_boundaries['ymin'])
                ymax = max(ymax, image_boundaries['ymax'])
            else:
                xmin = image_boundaries['xmin']
                xmax = image_boundaries['xmax']
                ymin = image_boundaries['ymin']
                ymax = image_boundaries['ymax']
        return {'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax}

    def T_estimate_values(self):
        """
        compute estimate values of tie points coordinates
        using geometric forward intersection
        :return: estimate values of tie points coordinates
        :rtype: np.array nx4: [point number,X,Y,Z]
        """
        # finding images with same tie points
        # and compute forward intersection to get estimate coordinates
        T_estimate_coordinates = np.zeros((int(max(self.__tie_points[:, 1])), 4))
        for i, img1 in enumerate(self.__images):
            for img2 in self.__images[i + 1:]:
                im_pair = ImagePair(img1, img2)

                # finding shared points
                xy, img1_ind, img2_ind = np.intersect1d(img1.T_samples[:, 0], img2.T_samples[:, 0], return_indices=True)

                # forward intersection
                T_estimate_coordinates_temp = (np.hstack((np.reshape(xy, (len(xy), 1)),
                                                          im_pair.geometricIntersection(img1.T_samples[img1_ind, 1:],
                                                                                        img2.T_samples[img2_ind, 1:]))))
                T_estimate_coordinates[np.uint32(T_estimate_coordinates_temp[:, 0] - 1)] = T_estimate_coordinates_temp
        self.T_coordinates = T_estimate_coordinates

    def BundleAdjustment(self, epsilon, max_itr):
        """
        BundleAdjustment for images block
        :param epsilon: stoping condition for norm(dx)
        :param max_itr: maximum number of iterations
        :return: exterior orientation parameters, tie points coordinate, RMSE, covariance matrix
        :rtype: np.array nx1, scalar, np.array mxm
        """

        points_coordinate = np.vstack((self.images[0].T_samples,self.images[0].GC_samples))

        # creating lb vector
        lb = self.create_lb_vector()

        dx = np.ones([6, 1]) * 100000
        itr = 0
        while la.norm(dx) > epsilon and itr < max_itr:
            itr = itr+1
            X = self.compute_variables_vector()
            l0 = self.compute_observation_vector()
            L = lb - l0
            A = self.ComputeDesignMatrix()
            U = np.dot(A.T, L)
            N = np.dot(A.T, A)

            # Schur Complement
            N11 = N[:4*len(self.images),:4*len(self.images)]
            N12 = N[:4*len(self.images),4*len(self.images):]
            N21 = N[4*len(self.images):,:4*len(self.images)]
            N22 = N[4*len(self.images):,4*len(self.images):]
            u1 = U[:4 * len(self.images)]
            u2 = U[4 * len(self.images):]
            N22_inv = la.inv(N22)
            dx_o = np.dot(la.inv(N11-N12.dot(N22_inv).dot(N12.T)),(u1-N12.dot(N22_inv).dot(u2)))
            dx_t = np.dot(N22_inv,(u2-np.dot(N12.T,dx_o)))
            dx = np.hstack((dx_o,dx_t))
            # plotting normal matrix
            # plt.figure()
            # plt.spy(N)
            # plt.title('N matrix')
            # plt.figure()
            # plt.subplot(221)
            # plt.spy(N11)
            # plt.title('N11')
            # plt.subplot(222)
            # plt.spy(N12)
            # plt.title('N12')
            # plt.subplot(223)
            # plt.spy(N21)
            # plt.title('N21')
            # plt.subplot(224)
            # plt.spy(N22)
            # plt.title('N22 ')
            # plt.show()
            # dx = np.dot(la.inv(N), U)
            X = X + dx
            v = A.dot(dx) - L
            RMSE = np.sqrt(np.dot(v.T,v)/(A.shape[0]-A.shape[1]))

            # updatind tie points values and exteriorOrientationParameters
            for i, im in enumerate(self.images):
                im.exteriorOrientationParameters[[0,1,2,5]] = X[4*i:4*i+4]
            self.T_coordinates[:,1:] = np.reshape(X[4*len(self.images):],(len(self.T_coordinates),3))

        sigmaX = RMSE**2 * (np.linalg.inv(N))
        return X,RMSE,sigmaX



    # ---------------------- Private methods ----------------------

    def ComputeDesignMatrix(self):
        """
            Compute the derivatives of the collinear law (design matrix)

            :return: The design matrix

            :rtype: np.array (2xsamples)x(4 x images number + tie points number x3)

        """
        for i, im in enumerate(self.images):

            # initialization for readability
            omega = im.exteriorOrientationParameters[3]
            phi = im.exteriorOrientationParameters[4]
            kappa = im.exteriorOrientationParameters[5]

            # Coordinates subtraction
            points = np.vstack((self.T_coordinates[np.uint32(im.T_samples[:,0])-1],self.GC_coordinates[np.uint32(im.GC_samples[:,0])-1]))
            dX = points[:, 1] - im.exteriorOrientationParameters[0]
            dY = points[:, 2] - im.exteriorOrientationParameters[1]
            dZ = points[:, 3] - im.exteriorOrientationParameters[2]
            dXYZ = np.vstack([dX, dY, dZ])

            rotationMatrixT = im.rotationMatrix.T
            rotatedG = rotationMatrixT.dot(dXYZ)
            rT1g = rotatedG[0, :]
            rT2g = rotatedG[1, :]
            rT3g = rotatedG[2, :]

            focalBySqauredRT3g = im.camera.focalLength / rT3g ** 2

            dxdg = rotationMatrixT[0, :][None, :] * rT3g[:, None] - rT1g[:, None] * rotationMatrixT[2, :][None, :]
            dydg = rotationMatrixT[1, :][None, :] * rT3g[:, None] - rT2g[:, None] * rotationMatrixT[2, :][None, :]

            dgdX0 = np.array([-1, 0, 0], 'f')
            dgdY0 = np.array([0, -1, 0], 'f')
            dgdZ0 = np.array([0, 0, -1], 'f')

            # Derivatives with respect to X0
            dxdX0 = -focalBySqauredRT3g * np.dot(dxdg, dgdX0)
            dydX0 = -focalBySqauredRT3g * np.dot(dydg, dgdX0)

            # Derivatives with respect to Y0
            dxdY0 = -focalBySqauredRT3g * np.dot(dxdg, dgdY0)
            dydY0 = -focalBySqauredRT3g * np.dot(dydg, dgdY0)

            # Derivatives with respect to Z0
            dxdZ0 = -focalBySqauredRT3g * np.dot(dxdg, dgdZ0)
            dydZ0 = -focalBySqauredRT3g * np.dot(dydg, dgdZ0)

            dgdX = np.array([1, 0, 0], 'f')
            dgdY = np.array([0, 1, 0], 'f')
            dgdZ = np.array([0, 0, 1], 'f')

            # Derivatives with respect to X
            dxdX = -focalBySqauredRT3g * np.dot(dxdg, dgdX)
            dydX = -focalBySqauredRT3g * np.dot(dydg, dgdX)

            # Derivatives with respect to Y
            dxdY = -focalBySqauredRT3g * np.dot(dxdg, dgdY)
            dydY = -focalBySqauredRT3g * np.dot(dydg, dgdY)

            # Derivatives with respect to Z
            dxdZ = -focalBySqauredRT3g * np.dot(dxdg, dgdZ)
            dydZ = -focalBySqauredRT3g * np.dot(dydg, dgdZ)

            # dRTdOmega = Compute3DRotationDerivativeMatrix(omega, phi, kappa, 'omega').T
            # dRTdPhi = Compute3DRotationDerivativeMatrix(omega, phi, kappa, 'phi').T
            dRTdKappa = Compute3DRotationDerivativeMatrix(omega, phi, kappa, 'kappa').T

            gRT3g = dXYZ * rT3g

            # Derivatives with respect to Omega
            # dxdOmega = -focalBySqauredRT3g * (dRTdOmega[0, :][None, :].dot(gRT3g) -
            #                                   rT1g * (dRTdOmega[2, :][None, :].dot(dXYZ)))[0]
            #
            # dydOmega = -focalBySqauredRT3g * (dRTdOmega[1, :][None, :].dot(gRT3g) -
            #                                   rT2g * (dRTdOmega[2, :][None, :].dot(dXYZ)))[0]
            #
            # # Derivatives with respect to Phi
            # dxdPhi = -focalBySqauredRT3g * (dRTdPhi[0, :][None, :].dot(gRT3g) -
            #                                 rT1g * (dRTdPhi[2, :][None, :].dot(dXYZ)))[0]
            #
            # dydPhi = -focalBySqauredRT3g * (dRTdPhi[1, :][None, :].dot(gRT3g) -
            #                                 rT2g * (dRTdPhi[2, :][None, :].dot(dXYZ)))[0]

            # Derivatives with respect to Kappa
            dxdKappa = -focalBySqauredRT3g * (dRTdKappa[0, :][None, :].dot(gRT3g) -
                                              rT1g * (dRTdKappa[2, :][None, :].dot(dXYZ)))[0]

            dydKappa = -focalBySqauredRT3g * (dRTdKappa[1, :][None, :].dot(gRT3g) -
                                              rT2g * (dRTdKappa[2, :][None, :].dot(dXYZ)))[0]

            # all derivatives of x and y
            dd1 = np.array([np.vstack([dxdX0, dxdY0, dxdZ0, dxdKappa]).T,
                           np.vstack([dydX0, dydY0, dydZ0, dydKappa]).T])
            dd2 = np.array([np.vstack([dxdX, dxdY, dxdZ]).T,
                           np.vstack([dydX, dydY, dydZ]).T])

            a1 = np.zeros((2 * dd1[0].shape[0], 4*len(self.images)))
            a2 = np.zeros((2 * dd2[0].shape[0], 3*len(self.T_coordinates)))
            a1[0::2,i*4:i*4+4] = dd1[0]
            a1[1::2,i*4:i*4+4] = dd1[1]
            for row in range(len(im.T_samples)):
                col = (int(im.T_samples[row,0])-1)*3
                a2[row*2,col:col+3] = dd2[0,row]
                a2[row*2+1,col:col+3] = dd2[1,row]
            a = np.hstack((a1,a2))
            if i == 0:
                A = a
            else:
                A = np.vstack((A,a))
        return A

    def compute_observation_vector(self):
        """
        Compute observation vector for solving the exterior orientation parameters of a single image
        based on their approximate values
        :return: observation vector
        :rtype: np.array  (2 x samples)x1
        """
        for i, im in enumerate(self.images):
            points_coordinate = np.vstack((self.T_coordinates[np.uint32(im.T_samples[:,0])-1],self.GC_coordinates[np.uint32(im.GC_samples[:,0])-1]))
            l0_temp = im.ComputeObservationVector(points_coordinate[:,1:])
            if i == 0:
                l0 = l0_temp
            else:
                l0 = np.hstack((l0,l0_temp))
        return l0

    def compute_variables_vector(self):
        """

        :return:
        :rtype: np.array (4 x images number + tie points number x3)x1
        """
        OrientationParameters = self.images[0].exteriorOrientationParameters[:4]
        for i, im in enumerate(self.images[1:]):
            OrientationParameters = np.hstack((OrientationParameters,im.exteriorOrientationParameters[:4]))

        return np.hstack((OrientationParameters,np.ravel(self.T_coordinates[:,1:])))

    def create_lb_vector(self):
        lb = np.hstack((np.ravel(self.images[0].T_samples[:,1:]),np.ravel(self.images[0].GC_samples[:,1:])))
        for i, im in enumerate(self.images[1:]):
            lb =np.hstack((lb,np.ravel(im.T_samples[:,1:]),np.ravel(im.GC_samples[:,1:])))
        return lb

    def draw_block(self, anotate=False):
        """
        drawing the images block in 2d
        :return: none
        """
       # create figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        for img in self.images:
            img.draw_frame(ax)
            img.draw_tie_points(ax, anotate=anotate)
            img.draw_control_points(ax, anotate=anotate)
            
        return ax
    
    def describe_block(self):
        # create data frame for images EOP
        EOP = pd.DataFrame(columns=['Image', 'X0', 'Y0', 'Z0', 'omega', 'phi', 'kappa'])
        for img in self.images:
            EOP.loc[len(EOP)] = [img.name] + list(img.exteriorOrientationParameters)
        
        print('Images EOP:')
        print(EOP)
        
        print('\nTie points:')
        print(self.tie_points)
        
        print('\nControl points:')
        print(self.control_points)
        

