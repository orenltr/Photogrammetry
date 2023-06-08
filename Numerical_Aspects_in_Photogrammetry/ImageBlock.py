from Camera import Camera
from SingleImage import SingleImage
from MatrixMethods import Compute3DRotationMatrix, Compute3DRotationDerivativeMatrix, ComputeSkewMatrixFromVector
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import linalg as la


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
        # self.block_points = self.block_points()  


    # ---------------------- Properties ----------------------
    
    # create property for tie points
    
    @property
    def tie_points(self):
        return self.__tie_points
    
    @tie_points.setter
    def tie_points(self, val):
        self.__tie_points = val
        self.update_images()
       
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
        self.update_images()
              
    
    @property
    def T_coordinates(self):
        """
        Tie points coordinates

        :return: Tie points coordinates

        :rtype: np.array nx4: [point number,X,Y,Z]

        """
        return self.tie_points[['X', 'Y', 'Z']].values

    @T_coordinates.setter
    def T_coordinates(self, val):
        """
        Tie points coordinates

        :param val: Tie points coordinates

        :type: np.array nx4: [point number,X,Y,Z]

        """

        self.tie_points[['X', 'Y', 'Z']] = val        
        
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
    
    @property
    def X(self):
        return self.compute_variables_vector()
    
    @property
    def A(self):
        return self.ComputeDesignMatrix()
    
    @property
    def N(self):        
        A = self.A
        N = np.dot(A.T, A)
        return N
    
    # ---------------------- methods ----------------------
    
    
    # merge tie points and control points of all images to one data frame
    @property
    def block_points(self):
        # get tie points and control points of all images
        for image in self.images:
            if 'block_points' in locals():
                temp = pd.concat([image.tie_points, image.control_points])
                temp['image_id'] = image.name
                block_points = pd.concat([block_points, temp])
            else:
                block_points = pd.concat([image.tie_points, image.control_points])
                block_points['image_id'] = image.name
        block_points = block_points.reset_index(drop=True)
            
        return block_points


    def BundleAdjustment(self, epsilon, max_itr, method='naive', **kwargs):
        """
        BundleAdjustment for images block
        :param epsilon: stoping condition for norm(dx)
        :param max_itr: maximum number of iterations
        :param method: method for solving the normal equation (naive, schur)
        :return: exterior orientation parameters, tie points coordinate, RMSE
        :rtype: np.array nx1, scalar, np.array mxm
        """

        # creating lb vector
        lb = self.create_lb_vector()
        
        # create vaiables to track the values of the variables
        # create a data frame with columns=['X0', 'Y0', 'Z0', 'omega', 'phi', 'kappa'] for each image and tie points coordinates and iteration number
        variables = pd.DataFrame(columns=['X0', 'Y0', 'Z0', 'omega', 'phi', 'kappa']*len(self.images)+['X', 'Y', 'Z']*len(self.T_coordinates))
        observation = pd.DataFrame(columns=['x', 'y']*int(len(lb)/2))

        dx = np.ones([6, 1]) * 100000
        itr = 0
        while la.norm(dx) > epsilon and itr < max_itr:
            itr += 1
            X = self.compute_variables_vector()
            l0 = self.compute_observation_vector()
            L = lb - l0
            A = self.ComputeDesignMatrix()
            
            # convert all to float64
            lb = lb.astype(np.float64)
            X = X.astype(np.float64)
            l0 = l0.astype(np.float64)
            L = L.astype(np.float64)
            A = A.astype(np.float64)
            
            # update variables data frame
            variables.loc[len(variables)] = np.ravel(X)
            # update observation data frame
            observation.loc[len(observation)] = np.ravel(l0)
            
            # # print report for iteration of A, X, L statistics
            # print('------------------------------------------------------------------------------')
            # print('iteration: ', itr, '\n norm(A): ', la.norm(A), 'norm(X): ', la.norm(X), 'norm(L): ', la.norm(L),'\n')
            # # print report for iteration of exterior orientation parameters and camera coordinates
            # print('EOP: ', X[:6*len(self.images)],'\n')
            
            if method == 'naive':
                U = np.dot(A.T, L)
                N = np.dot(A.T, A)
                dx = np.dot(la.inv(N), U)
                
                if 'plotNormal' in kwargs:
                    if kwargs['plotNormal'] :
                        # # plotting normal matrix
                        ImageBlock.plotNormalMatrix(N)
                        
                
                # Compute the singular value decomposition of A (SVD)
                U, s, V = np.linalg.svd(A)
                # Compute the condition number of A
                cond_A = np.max(s) / np.min(s)
                
                # Compute the singular value decomposition of N (SVD)
                U, s, V = np.linalg.svd(N)
                # Compute the condition number of A
                cond_N = np.max(s) / np.min(s)
               
                print('iteration: ', itr, '\n condition number A: ', cond_A)
                print(' condition number N: ', cond_N, '\n')                
                    
            elif method == 'schur':                  
                # sparse matrix with Schur Complement
                N11 = self.ComputeN11Block(A)
                N22 = self.ComputeN22Block(A)
                N12 = self.ComputeN12Block(A)     
                u1 = self.ComputeU1Block(A, L)
                u2 = self.ComputeU2Block(A, L)  
                N22_inv = la.inv(N22)
                dx_o = np.dot(la.inv(N11-N12.dot(N22_inv).dot(N12.T)),(u1-N12.dot(N22_inv).dot(u2)))
                dx_t = np.dot(N22_inv,(u2-np.dot(N12.T,dx_o)))
                dx = np.hstack((dx_o,dx_t))         
                
                if 'plotNormal' in kwargs:
                    if kwargs['plotNormal'] :
                        # # plotting normal matrix
                        ImageBlock.plotNormalBlocks(N11, N12, N22)
                                
            else:
                raise ValueError('method should be naive or schur')
            
            # update variables
            X = X + dx
            v = A.dot(dx) - L
            
            # updatind tie points values and exteriorOrientationParameters
            for i, im in enumerate(self.images):
                im.exteriorOrientationParameters = X[6*i:6*i+6]
            self.T_coordinates = np.reshape(X[6*len(self.images):],(len(self.T_coordinates),3))
            
            # calculate RMSE
            RMSE = np.sqrt(np.dot(v.T,v)/(A.shape[0]-A.shape[1]))            
            # print('RMSE: ', RMSE, 'norm(dx): ', la.norm(dx), '\n')
            
            sigmaX = RMSE**2 * (np.linalg.inv(N))            

        return X,RMSE


    
    def plotNormalBlocks(N11, N12, N22):
        # N11 = N[:6*len(self.images),:6*len(self.images)]
        # N12 = N[:6*len(self.images),6*len(self.images):]
        # N21 = N[6*len(self.images):,:6*len(self.images)]
        # N22 = N[6*len(self.images):,6*len(self.images):]
        # plt.figure()
        # plt.spy(N)
        # plt.title('N matrix')
        plt.figure(figsize=(10,10))
        plt.subplot(221)
        plt.spy(N11)
        plt.title('N11')
        plt.subplot(222)
        plt.spy(N12)
        plt.title('N12')
        plt.subplot(223)
        plt.spy(N12.T)
        plt.title('N21')
        plt.subplot(224)
        plt.spy(N22)
        plt.title('N22 ')
        plt.show()
        
        
    def ComputeU1Image(self, Ai, image, Li):
        """ Compute U1 matrix for a single image
        input:
            Ai: design matrix for a single image (2*number of ground points, 6)
            image: image object
        output:
            U1i: U1 matrix for a single image (6,6)
        """

        U1i = sum(map(lambda i: Ai[2*i:2*i+2, :].T @ Li[2*i:2*i+2], range(image.ground_coords.shape[0])))
        return U1i

    def ComputeU1Block(self, A, L):
        
        # creating lists of rows and cloumns indices
        A_indices_rows = np.hstack((0,np.cumsum([img.ground_coords.shape[0] * 2 for img in self.images])))    
        A_indices_cols = np.arange(0,len(self.images)*6+1,6)
        # slicing A matrix for computing N11i blocks
        A_slices = list(map(lambda r1,r2,c1,c2: A[r1:r2,c1:c2],A_indices_rows[:-1],A_indices_rows[1:],A_indices_cols[:-1],A_indices_cols[1:]))
        
        L_slices = list(map(lambda r1,r2: L[r1:r2],A_indices_rows[:-1],A_indices_rows[1:]))
        # computing N11i blocks
        U1i_blocks = list(map(lambda x: self.ComputeU1Image(*x), zip(A_slices, self.images, L_slices)))
        
        U1 = np.hstack(U1i_blocks).T     
        return U1
    
    def ComputeU2Block(self, A, L):
        U2k_blocks = []
        # compute U2 for each tie point - variable index
        U2k_blocks = self.block_points.groupby('tie_block_id').apply(self.ComputeU2k, A=A, L=L)
            
        return np.hstack(U2k_blocks).T

    def ComputeU2k(self, group, A, L):
        # Access the original indices and image_ids of the group - observation indices
        original_indices = group.index.values  #   
        image_ids = group['image_id'].values    
        
        U2k = sum(map(lambda i,j: -A[j*2:j*2+2, 6*image_ids[i]:6*image_ids[i]+3].T @ L[j*2:j*2+2], np.arange(len(original_indices)),original_indices))

        return U2k 
        
    
    # build Normal matrix blocks
    def ComputeN11Image(self, Ai, image):
        """ Compute N11 matrix for a single image
        input:
            Ai: design matrix for a single image (2*number of ground points, 6)
            image: image object
        output:
            N11i: N11 matrix for a single image (6,6)
        """

        N11i = sum(map(lambda i: Ai[2*i:2*i+2, :].T @ Ai[2*i:2*i+2, :], range(image.ground_coords.shape[0])))
        return N11i

    def ComputeN11Block(self, A):
        
        # creating lists of rows and cloumns indices
        A_indices_rows = np.hstack((0,np.cumsum([img.ground_coords.shape[0] * 2 for img in self.images])))    
        A_indices_cols = np.arange(0,len(self.images)*6+1,6)
        # slicing A matrix for computing N11i blocks
        A_slices = list(map(lambda r1,r2,c1,c2: A[r1:r2,c1:c2],A_indices_rows[:-1],A_indices_rows[1:],A_indices_cols[:-1],A_indices_cols[1:]))
        # computing N11i blocks
        N11i_blocks = list(map(lambda x: self.ComputeN11Image(*x), zip(A_slices, self.images)))
        
        # plot N11i blocks
        plt.figure(figsize=(10,10))
        for i, N11i in enumerate(N11i_blocks):
            plt.subplot(3, 3, i+1)
            plt.spy(N11i)
            plt.title('N11i'+str(i))
        plt.tight_layout()
        plt.show()
        
        
        N11 = la.block_diag(*N11i_blocks)     
        return N11
        
    def ComputeN22Block(self, A):
        N22k_blocks = []
        # compute N22 for each tie point - variable index
        N22k_blocks = self.block_points.groupby('tie_block_id').apply(self.ComputeN22k, A=A)
            
        # plot N22k blocks
        plt.figure(figsize=(10,10))
        for i, N22k in enumerate(N22k_blocks):
            plt.subplot(3, 3, i+1)
            plt.spy(N22k)
            plt.title('N22k'+str(i))
        plt.tight_layout()
        plt.show()
        
        return la.block_diag(*N22k_blocks)

    def ComputeN22k(self, group, A):
        # Access the original indices and image_ids of the group - observation indices
        original_indices = group.index.values  #   
        image_ids = group['image_id'].values    
        
        N22k = sum(map(lambda i,j: -A[j*2:j*2+2, 6*image_ids[i]:6*image_ids[i]+3].T @ -A[j*2:j*2+2, 6*image_ids[i]:6*image_ids[i]+3], np.arange(len(original_indices)),original_indices))

        return N22k 

    def ComputeN12Block(self, A):
            
        # creating lists of rows and cloumns indices
        A_indices_rows = np.hstack((0,np.cumsum([img.ground_coords.shape[0] * 2 for img in self.images])))    
        A_indices_cols = np.arange(0,len(self.images)*6+1,6)
        # slicing A matrix for computing N11i blocks
        A_slices = list(map(lambda r1,r2,c1,c2: A[r1:r2,c1:c2],A_indices_rows[:-1],A_indices_rows[1:],A_indices_cols[:-1],A_indices_cols[1:]))
        # computing N12 blocks
        N12_blocks = list(map(lambda x: self.ComputeN12i(*x), zip(A_slices, self.images)))    
            
        # plot N12 blocks
        plt.figure(figsize=(10,10))
        for i, N12 in enumerate(N12_blocks):
            plt.subplot(3, 3, i+1)
            plt.spy(N12)
            plt.title('N12'+str(i))
        plt.tight_layout()
        plt.show()
        
        return np.vstack(N12_blocks)

    def ComputeN12i(self, Ai, img):
    
        observation_indices_in_img = img.tie_points['tie_block_id'].values    
            
        def populateN12i(i):
            if i in observation_indices_in_img:            
                i = np.where(observation_indices_in_img == i)[0][0] 
                Aij = Ai[i*2:i*2+2, :6]
                Bjk = -Ai[i*2:i*2+2, :3]
                N12i = Aij.T @ Bjk
            else:            
                N12i = np.zeros((6,3))
                        
            return N12i
            
        # # temp_populateN12i = partial(populateN12i,N12i)
        N12i = np.hstack(list(map(populateN12i, np.arange(self.tie_points.shape[0]))))
            
        return N12i
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
            points = im.ground_coords
            # points = np.vstack((self.T_coordinates[np.uint32(im.T_samples[:,0])-1],self.GC_coordinates[np.uint32(im.GC_samples[:,0])-1]))
            dX = points[:, 0] - im.exteriorOrientationParameters[0]
            dY = points[:, 1] - im.exteriorOrientationParameters[1]
            dZ = points[:, 2] - im.exteriorOrientationParameters[2]
            dXYZ = np.vstack([dX, dY, dZ])

            rotationMatrixT = im.rotationMatrix.T
            rotatedG = rotationMatrixT.dot(dXYZ)
            rT1g = rotatedG[0, :]
            rT2g = rotatedG[1, :]
            rT3g = rotatedG[2, :]

            focalBySqauredRT3g = im.camera.focal_length / rT3g ** 2

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

            dRTdOmega = Compute3DRotationDerivativeMatrix(omega, phi, kappa, 'omega').T
            dRTdPhi = Compute3DRotationDerivativeMatrix(omega, phi, kappa, 'phi').T
            dRTdKappa = Compute3DRotationDerivativeMatrix(omega, phi, kappa, 'kappa').T

            gRT3g = dXYZ * rT3g

            # Derivatives with respect to Omega
            dxdOmega = -focalBySqauredRT3g * (dRTdOmega[0, :][None, :].dot(gRT3g) -
                                              rT1g * (dRTdOmega[2, :][None, :].dot(dXYZ)))[0]
            
            dydOmega = -focalBySqauredRT3g * (dRTdOmega[1, :][None, :].dot(gRT3g) -
                                              rT2g * (dRTdOmega[2, :][None, :].dot(dXYZ)))[0]
            
            # Derivatives with respect to Phi
            dxdPhi = -focalBySqauredRT3g * (dRTdPhi[0, :][None, :].dot(gRT3g) -
                                            rT1g * (dRTdPhi[2, :][None, :].dot(dXYZ)))[0]
            
            dydPhi = -focalBySqauredRT3g * (dRTdPhi[1, :][None, :].dot(gRT3g) -
                                            rT2g * (dRTdPhi[2, :][None, :].dot(dXYZ)))[0]

            # Derivatives with respect to Kappa
            dxdKappa = -focalBySqauredRT3g * (dRTdKappa[0, :][None, :].dot(gRT3g) -
                                              rT1g * (dRTdKappa[2, :][None, :].dot(dXYZ)))[0]

            dydKappa = -focalBySqauredRT3g * (dRTdKappa[1, :][None, :].dot(gRT3g) -
                                              rT2g * (dRTdKappa[2, :][None, :].dot(dXYZ)))[0]

            # all derivatives of x and y
            dd1 = np.array([np.vstack([dxdX0, dxdY0, dxdZ0, dxdOmega, dxdPhi, dxdKappa]).T,
                           np.vstack([dydX0, dydY0, dydZ0, dydOmega, dydPhi, dydKappa]).T])
            dd2 = np.array([np.vstack([dxdX, dxdY, dxdZ]).T,
                           np.vstack([dydX, dydY, dydZ]).T])

            # divide A matrix to 2 parts: for EOP and for tie points
            a1 = np.zeros((2 * dd1[0].shape[0], 6*len(self.images)))
            a2 = np.zeros((2 * dd2[0].shape[0], 3*len(self.T_coordinates)))
            
            # populate EOP derivatives 
            a1[0::2,i*6:i*6+6] = dd1[0]
            a1[1::2,i*6:i*6+6] = dd1[1]
            
            # populate tie points derivatives
            for row in range(len(im.tie_points)):
                # find the column of the tie point in A matrix
                col = (int(im.tie_points['tie_block_id'][row]))*3
                # populate derivatives
                a2[row*2,col:col+3] = dd2[0,row]
                a2[row*2+1,col:col+3] = dd2[1,row]
            # combine A matrix
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
            l0_temp = im.ComputeObservationVector()
            if i == 0:
                l0 = l0_temp
            else:
                l0 = np.hstack((l0,l0_temp))
        return l0

    def compute_variables_vector(self):
        """

        :return:
        :rtype: np.array (6 x images number + tie points number x3)x1
        """
        OrientationParameters = self.images[0].exteriorOrientationParameters
        for i, im in enumerate(self.images[1:]):
            OrientationParameters = np.hstack((OrientationParameters,im.exteriorOrientationParameters))

        return np.hstack((OrientationParameters,np.ravel(self.T_coordinates)))

    def create_lb_vector(self):
        
        # initialize lb vector with the first image tie points and control points
        lb =np.hstack((np.ravel(self.images[0].tie_points_cam_coords.T),np.ravel(self.images[0].control_points_cam_coords.T)))
        # add tie points and control points of the rest of the images
        for i, im in enumerate(self.images[1:]):
            lb =np.hstack((lb,np.ravel(im.tie_points_cam_coords.T),np.ravel(im.control_points_cam_coords.T)))            
        return lb

    def draw_block(self, anotate=False ,ax=None):
        """
        drawing the images block in 2d
        :return: none
        """
       # create figure
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        for img in self.images:
            img.draw_frame(ax)        
            
        for img in self.images:
            img.draw_tie_points(ax, anotate=anotate)
            img.draw_control_points(ax, anotate=anotate)
        
        return ax
    
    def describe_block(self):
        # create data frame for images EOP
        EOP = pd.DataFrame(columns=['Image', 'X0', 'Y0', 'Z0', 'omega', 'phi', 'kappa'])
        for img in self.images:
            # convert exterior orientation parameters to arc seconds
            rotations = np.rad2deg(img.exteriorOrientationParameters[3:]) * 3600
            EOP.loc[len(EOP)] = [img.name] + list(img.exteriorOrientationParameters[:3]) + list(rotations)
        
        print('Images EOP:')
        print(EOP)
        
        print('\nTie points:')
        print(self.tie_points)
        
        print('\nControl points:')
        print(self.control_points)
        
        return EOP
        
    def update_images(self):
        # update images tie points and control points coordinates from block.tie_points and block.control_points
        for img in self.images:
            img.tie_points[['x','y']] = self.tie_points[self.tie_points.index.isin(img.tie_points['tie_block_id'])][['x','y']].values
            img.tie_points[['X','Y','Z']] = self.tie_points[self.tie_points.index.isin(img.tie_points['tie_block_id'])][['X','Y','Z']].values
            if 'x' in self.control_points.columns:
                img.control_points[['x','y']] = self.control_points[self.control_points.name.isin(img.control_points['name'])][['x','y']].values
            img.control_points[['X','Y','Z']] = self.control_points[self.control_points.name.isin(img.control_points['name'])][['X','Y','Z']].values
    
    
        
            
            
        

