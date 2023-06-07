import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt
from Camera import *
from MatrixMethods import Compute3DRotationMatrix, Compute3DRotationDerivativeMatrix, Compute3DRotationMatrix_RzRyRz, \
    Compute3DRotationDerivativeMatrix_RzRyRz


class SingleImage(object):

    def __init__(self, camera, name, exteriorOrientationParameters=np.array([0,0,0,0,0,0]) , tie_points=None, control_points=None):
        """
        Initialize the SingleImage object

        :param camera: instance of the Camera class
        :param points: points in image space

        :type camera: Camera
        :type points: np.array

        """
        self.__camera = camera
        self.name = name
        self.tie_points = tie_points # data frame with columns=['x', 'y', 'name', 'image_id', 'X', 'Y', 'Z']
        self.control_points = control_points # data frame with columns=['x', 'y', 'name', 'image_id', 'X', 'Y', 'Z']
        self.__innerOrientationParameters = None
        self.__exteriorOrientationParameters = exteriorOrientationParameters    # np.array([X0,Y0,Z0,omega,phi,kappa])
        self.__rotationMatrix = None
        
        
    # property to get tie points as a numpy array
    @property
    def tie_points_cam_coords(self): 
        points = self.tie_points[['x', 'y']].values
        return points.T
    
    @property
    def tie_points_ground_coords(self):         
        return self.tie_points[['X', 'Y', 'Z']].values
    
    @property
    def control_points_cam_coords(self): 
        points = self.control_points[['x', 'y']].values
        return points.T
    
    @property
    def control_points_ground_coords(self):
        return self.control_points[['X', 'Y', 'Z']].values
    
    @property
    def ground_coords(self): 
        # create matrix with ground coordinates of tie points and control points
        ground_coords = np.vstack((self.tie_points[['X', 'Y', 'Z']].values, self.control_points[['X', 'Y', 'Z']].values))        
        return ground_coords
    
    @property
    def num_tie_points(self):
        return len(self.tie_points)
    
    @property
    def num_control_points(self):
        return len(self.control_points)
    
    # property to get image ground bounds
    @property
    def image_ground_bounds(self):
        ground_points = self.frame_to_ground()
        # create dictionary with bounds
        bounds = {'xmin': np.min(ground_points[0,:]), 'xmax': np.max(ground_points[0,:]),
                    'ymin': np.min(ground_points[1,:]), 'ymax': np.max(ground_points[1,:])}
        return bounds
    
    @property
    def scale(self):
        """
        Scale of the image

        :return: scale of the image

        :rtype: float
        """
        alt = self.exteriorOrientationParameters[2]
        f = self.camera.focal_length
        return f / alt
    
    @property
    def innerOrientationParameters(self):
        """
        Inner orientation parameters


        .. warning::

            Can be held either as dictionary or array. For your implementation and decision.

        .. note::

            Do not forget to decide how it is held and document your decision

        :return: inner orinetation parameters

        :rtype: **ADD**
        """
        return self.__innerOrientationParameters

    @property
    def camera(self):
        """
        The camera that took the image

        :rtype: Camera

        """
        return self.__camera

    @property
    def exteriorOrientationParameters(self):
        r"""
        Property for the exterior orientation parameters

        :return: exterior orientation parameters in the following order, **however you can decide how to hold them (dictionary or array)**

        .. math::
            exteriorOrientationParameters = \begin{bmatrix} X_0 \\ Y_0 \\ Z_0 \\ \omega \\ \varphi \\ \kappa \end{bmatrix}

        :rtype: np.ndarray or dict
        """
        return self.__exteriorOrientationParameters

    @exteriorOrientationParameters.setter
    def exteriorOrientationParameters(self, parametersArray):
        r"""

        :param parametersArray: the parameters to update the ``self.__exteriorOrientationParameters``

        **Usage example**

        .. code-block:: py

            self.exteriorOrintationParameters = parametersArray

        """
        self.__exteriorOrientationParameters = parametersArray

    @innerOrientationParameters.setter
    def innerOrientationParameters(self, parametersArray):
        r"""

        :param parametersArray: the parameters to update the ``self.__innerOrientationParameters``

        **Usage example**

        .. code-block:: py

            self.innerOrientationParameters = parametersArray

        """
        self.__innerOrientationParameters = parametersArray

    @property
    def rotationMatrix(self):
        """
        The rotation matrix of the image

        Relates to the exterior orientation
        :return: rotation matrix

        :rtype: np.ndarray (3x3)
        """

        return Compute3DRotationMatrix(*self.exteriorOrientationParameters[3:])

    @rotationMatrix.setter
    def rotationMatrix(self, R):
        """
        The rotation matrix of the image

        Relates to the exterior orientation
        :return: rotation matrix

        :rtype: np.ndarray (3x3)
        """

        self.__rotationMatrix = R

    @property
    def rotationMatrix_RzRyRz(self):
        """
        The rotation matrix of the image

        Relates to the exterior orientation
        :return: rotation matrix

        :rtype: np.ndarray (3x3)
        """

        R = Compute3DRotationMatrix_RzRyRz(self.exteriorOrientationParameters[3], self.exteriorOrientationParameters[4],
                                           self.exteriorOrientationParameters[5])

        return R

    
    def ComputeGeometricParameters(self):
        """
        Computes the geometric inner orientation parameters

        :return: geometric inner orientation parameters

        :rtype: dict

        .. warning::

           This function is empty, need implementation

        .. note::

            The algebraic inner orinetation paramters are held in ``self.innerOrientatioParameters`` and their type
            is according to what you decided when initialized them

        """
        # extracting inner orientation params
        a0 = self.innerOrientationParameters[0]
        b0 = self.innerOrientationParameters[1]
        a1 = self.innerOrientationParameters[2]
        a2 = self.innerOrientationParameters[3]
        b1 = self.innerOrientationParameters[4]
        b2 = self.innerOrientationParameters[5]

        # computing algebric params
        tx = a0
        ty = b0
        theta = np.arctan(b1 / b2)
        gamma = np.arctan((a1 * np.sin(theta) + a2 * np.cos(theta)) / (b1 * np.sin(theta) + b2 * np.cos(theta)))
        sx = a1 * np.cos(theta) - a2 * np.sin(theta)
        sy = (a1 * np.sin(theta) + a2 * np.cos(theta)) / np.sin(gamma)

        return {"translationX": tx, "translationY": ty, "rotationAngle": np.rad2deg(theta), "scaleFactorX": sx,
                "scaleFactorY": sy, "shearAngle": np.rad2deg(gamma)}

   
   

    def ComputeExteriorOrientation(self, imagePoints, groundPoints, epsilon):
        """
        Compute exterior orientation parameters.

        This function can be used in conjecture with ``self.__ComputeDesignMatrix(groundPoints)`` and ``self__ComputeObservationVector(imagePoints)``

        :param imagePoints: image points
        :param groundPoints: corresponding ground points

            .. note::

                Angles are given in radians

        :param epsilon: threshold for convergence criteria

        :type imagePoints: np.array nx2
        :type groundPoints: np.array nx3
        :type epsilon: float

        :return: Exterior orientation parameters: (X0, Y0, Z0, omega, phi, kappa), their accuracies, and residuals vector. *The orientation parameters can be either dictionary or array -- to your decision*

        :rtype: dict


        """
        # cameraPoints = self.ImageToCamera(imagePoints)
        cameraPoints = imagePoints
        self.__ComputeApproximateVals(cameraPoints, groundPoints)
        l0 = self.__ComputeObservationVector(groundPoints.T)
        l0 = np.reshape(l0, (-1, 1))
        l = cameraPoints.reshape(np.size(cameraPoints), 1) - l0
        A = self.__ComputeDesignMatrix(groundPoints.T)

        N = np.dot(A.T, A)
        u = np.dot(A.T, l)
        deltaX = np.dot(la.inv(N), u)

        # update orientation pars
        self.__exteriorOrientationParameters = np.add(self.__exteriorOrientationParameters, np.reshape(deltaX, 6))

        while la.norm(deltaX) > epsilon:
            l0 = self.__ComputeObservationVector(groundPoints.T)
            l0 = np.reshape(l0, (-1, 1))
            l = cameraPoints.reshape(np.size(cameraPoints), 1) - l0
            A = self.__ComputeDesignMatrix(groundPoints.T)
            N = np.dot(A.T, A)
            u = np.dot(A.T, l)
            deltaX = np.dot(la.inv(N), u)
            # update orientation pars
            self.__exteriorOrientationParameters = np.add(self.__exteriorOrientationParameters, np.reshape(deltaX, 6))

        # compute residuals
        l_a = np.reshape(self.__ComputeObservationVector(groundPoints.T), (-1, 1))
        v = l_a - cameraPoints.reshape(np.size(cameraPoints), 1)
        if (np.size(A, 0) - np.size(deltaX)) != 0:
            sig = np.dot(v.T, v) / (np.size(A, 0) - np.size(deltaX))
            sigmaX = sig[0] * la.inv(N)
        else:
            sigmaX = None

        return [self.exteriorOrientationParameters, sigmaX, v]


    def GroundToImage(self, groundPoints):
        """
        Transforming ground points to image points

        :param groundPoints: ground points [m]

        :type groundPoints: np.array nx3

        :return: corresponding Image points

        :rtype: np.array nx2

        """
        X0 = float(self.exteriorOrientationParameters[0])
        Y0 = float(self.exteriorOrientationParameters[1])
        Z0 = float(self.exteriorOrientationParameters[2])

        xp = float(self.camera.principalPoint[0])
        yp = float(self.camera.principalPoint[1])

        if self.camera.radialDistortions is not None:
            K1 = float(self.camera.radialDistortions['K1'])
            K2 = float(self.camera.radialDistortions['K2'])
        else:
            K1, K2 = 0, 0

        R = self.rotationMatrix.T
        r11 = float(R[0, 0])
        r12 = float(R[0, 1])
        r13 = float(R[0, 2])
        r21 = float(R[1, 0])
        r22 = float(R[1, 1])
        r23 = float(R[1, 2])
        r31 = float(R[2, 0])
        r32 = float(R[2, 1])
        r33 = float(R[2, 2])

        f = self.camera.focalLength

        camPoints = []

        for i in range(groundPoints.shape[0]):
            x = xp - (f) * (((r11 * (groundPoints[i, 0] - X0) + r21 * (groundPoints[i, 1] - Y0) + r31 * (
                    groundPoints[i, 2] - Z0)) / (r13 * (groundPoints[i, 0] - X0) + r23 * (
                    groundPoints[i, 1] - Y0) + r33 * (groundPoints[i, 2] - Z0))))

            y = yp - (f) * (((r12 * (groundPoints[i, 0] - X0) + r22 * (groundPoints[i, 1] - Y0) + r32 * (
                    groundPoints[i, 2] - Z0)) / (r13 * (groundPoints[i, 0] - X0) + r23 * (
                    groundPoints[i, 1] - Y0) + r33 * (groundPoints[i, 2] - Z0))))

            rr = np.sqrt((x - xp) ** 2 + (y - yp) ** 2)
            x = x + (x) * (K1 * rr ** 2 + K2 * rr ** 4)
            y = y + (y) * (K1 * rr ** 2 + K2 * rr ** 4)

            camPoints.append([x, y])

        # return self.CameraToImage(np.array(camPoints))
        return (np.array(camPoints))
    
    def GroundToImage_fast(self, groundPoints):
        # EOP
        X0 = float(self.exteriorOrientationParameters[0])
        Y0 = float(self.exteriorOrientationParameters[1])
        Z0 = float(self.exteriorOrientationParameters[2])
        R = self.rotationMatrix.T
        f = self.camera.focal_length
        
        # Coordinates subtraction
        dX = groundPoints[:, 0] - self.exteriorOrientationParameters[0]
        dY = groundPoints[:, 1] - self.exteriorOrientationParameters[1]
        dZ = groundPoints[:, 2] - self.exteriorOrientationParameters[2]
        dXYZ = np.vstack([dX, dY, dZ])

        rotationMatrixT = self.rotationMatrix.T
        rotatedG = rotationMatrixT.dot(dXYZ)
        rT1g = rotatedG[0, :]
        rT2g = rotatedG[1, :]
        rT3g = rotatedG[2, :]
        x = -f * rT1g / rT3g
        y = -f * rT2g / rT3g
        
        return np.vstack([x, y]).T


    def ImageToGround_GivenZ(self, imagePoints, Z_values):
        """
        Compute corresponding ground point given the height in world system

        :param imagePoints: points in image space
        :param Z_values: height of the ground points


        :type Z_values: np.array nx1
        :type imagePoints: np.array 2xn
        :type eop: np.ndarray 6x1

        :return: corresponding ground points

        :rtype: np.ndarray 3xn

        """
        cameraPoints = imagePoints
        omega = self.exteriorOrientationParameters[3]
        phi = self.exteriorOrientationParameters[4]
        kapa = self.exteriorOrientationParameters[5]
        X0 = self.exteriorOrientationParameters[0]
        Y0 = self.exteriorOrientationParameters[1]
        Z0 = self.exteriorOrientationParameters[2]

        R = Compute3DRotationMatrix(omega, phi, kapa)

        f = self.camera.focal_length

        # insert -f to the end of each point
        cameraPoints = np.vstack((cameraPoints, -f * np.ones((1, cameraPoints.shape[1]))))
        # calculating scale factor
        s = (Z_values - Z0) / (R[2, :].dot(cameraPoints))
        # creating [X0,Y0,Z0] vector
        camera_location = np.vstack((X0, Y0, Z0))
        # calculating ground points
        groundPoints = camera_location + s * R.dot(cameraPoints)
        
        return groundPoints

   
    def ComputeObservationVector(self):
        """
        Compute observation vector for solving the exterior orientation parameters of a single image
        based on their approximate values

        :param groundPoints: Ground coordinates of the control points

        :type groundPoints: np.array nx3

        :return: Vector l0

        :rtype: np.array nx1
        """
        groundPoints = self.ground_coords

        n = groundPoints.shape[0]  # number of points

        # Coordinates subtraction
        dX = groundPoints[:,0] - self.exteriorOrientationParameters[0]
        dY = groundPoints[:,1] - self.exteriorOrientationParameters[1]
        dZ = groundPoints[:,2] - self.exteriorOrientationParameters[2]
        dXYZ = np.vstack([dX, dY, dZ])
        rotated_XYZ = np.dot(self.rotationMatrix.T, dXYZ).T

        l0 = np.zeros(n * 2)

        # Computation of the observation vector based on approximate exterior orientation parameters:
        l0[::2] = -self.camera.focal_length * rotated_XYZ[:, 0] / rotated_XYZ[:, 2]
        l0[1::2] = -self.camera.focal_length * rotated_XYZ[:, 1] / rotated_XYZ[:, 2]

        return l0

    def __ComputeDesignMatrix(self, groundPoints):
        """
            Compute the derivatives of the collinear law (design matrix)

            :param groundPoints: Ground coordinates of the control points

            :type groundPoints: np.array nx3

            :return: The design matrix

            :rtype: np.array nx6

        """
        # initialization for readability
        omega = self.exteriorOrientationParameters[3]
        phi = self.exteriorOrientationParameters[4]
        kappa = self.exteriorOrientationParameters[5]

        # Coordinates subtraction
        dX = groundPoints[:, 0] - self.exteriorOrientationParameters[0]
        dY = groundPoints[:, 1] - self.exteriorOrientationParameters[1]
        dZ = groundPoints[:, 2] - self.exteriorOrientationParameters[2]
        dXYZ = np.vstack([dX, dY, dZ])

        rotationMatrixT = self.rotationMatrix.T
        rotatedG = rotationMatrixT.dot(dXYZ)
        rT1g = rotatedG[0, :]
        rT2g = rotatedG[1, :]
        rT3g = rotatedG[2, :]

        focalBySqauredRT3g = self.camera.focalLength / rT3g ** 2

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
        dd = np.array([np.vstack([dxdX0, dxdY0, dxdZ0, dxdOmega, dxdPhi, dxdKappa]).T,
                       np.vstack([dydX0, dydY0, dydZ0, dydOmega, dydPhi, dydKappa]).T])

        a = np.zeros((2 * dd[0].shape[0], 6))
        a[0::2] = dd[0]
        a[1::2] = dd[1]

        return a

    
    def frame_to_ground(self):
        
        # this section defines each point
        tl = np.array([[-self.camera.sensor_size[0]/2], [self.camera.sensor_size[1]/2]])  # top left point
        tr = np.array([[self.camera.sensor_size[0]/2], [self.camera.sensor_size[1]/2]])  # top right point
        bl = np.array([[-self.camera.sensor_size[0]/2], [-self.camera.sensor_size[1]/2]])  # bot left point
        br = np.array([[self.camera.sensor_size[0]/2], [-self.camera.sensor_size[1]/2]])  # bot right point

        # covert to 2x4 matrix
        frame_points = np.hstack((tl, tr, br, bl))
        
        # tranform to ground system
        ground_points = self.ImageToGround_GivenZ(frame_points, np.zeros(4))

        return ground_points
    
    def draw_frame(self,ax=[], anotate=False, color='random'):
        
        # project frame points to ground
        ground_points = self.frame_to_ground()
        
        # add first point to the end to close the frame
        ground_points = np.hstack((ground_points, ground_points[:,0].reshape(3,1)))
        
        # plot the frame corners
        ax.scatter(ground_points[0,:], ground_points[1,:], c='b', s=5)
        
        # choose color
        if color == 'random':
            red = np.random.randint(0, 255)
            green = np.random.randint(0, 255)
            blue = np.random.randint(0, 255)
            c = (red/255, green/255, blue/255)
        else:
            c = color
        
        ax.plot(ground_points[0,:], ground_points[1,:], color=c, label='images frames', linewidth=3)

        
    def draw_tie_points(self, ax=[], anotate=False):
        ax.scatter(self.tie_points['X'], self.tie_points['Y'], c='b', s=10*(1/self.scale), label='tie points')

        # anoate tie points using their names
        if anotate:
            self.tie_points.apply(lambda row: ax.annotate(row['name'], (row['X'], row['Y']), size=1*(1/self.scale)), axis=1)
    
    def draw_control_points(self, ax=[], anotate=False):
        ax.scatter(self.control_points['X'], self.control_points['Y'], marker='^', color='r', s=30*(1/self.scale) , label='GC')

        # anoate tie points using their names
        if anotate:
            self.control_points.apply(lambda row: ax.annotate(row['name'], (row['X'], row['Y']), size=1*(1/self.scale)), axis=1)
            
        

    def is_point_in_image(self, point):
        """
        Check if a point is in the image

        :param point: point in image space

        :type point: np.array 1x3

        :return: True if the point is in the image, False otherwise
        :rtype: bool
        """
        # check if the point is in the image ground boundaries
        if point[0]<self.image_ground_bounds['xmin'] or point[0]>self.image_ground_bounds['xmax'] or \
            point[1]<self.image_ground_bounds['ymin'] or point[1]>self.image_ground_bounds['ymax']:
            return False
        else:
            return True
        