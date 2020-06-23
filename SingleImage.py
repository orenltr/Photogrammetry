import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Camera import *
from Reader import *
from MatrixMethods import Compute3DRotationMatrix, Compute3DRotationDerivativeMatrix, Compute3DRotationMatrix_RzRyRz, \
    Compute3DRotationDerivativeMatrix_RzRyRz


class SingleImage(object):

    def __init__(self, camera):
        """
        Initialize the SingleImage object

        :param camera: instance of the Camera class
        :param points: points in image space

        :type camera: Camera
        :type points: np.array

        """
        self.__camera = camera
        self.__innerOrientationParameters = None
        self.__isSolved = False
        self.__exteriorOrientationParameters = np.array([0, 0, 0, 0, 0, 0], 'f')
        self.__rotationMatrix = None

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

        # R = Compute3DRotationMatrix(self.exteriorOrientationParameters[3], self.exteriorOrientationParameters[4],
        #                             self.exteriorOrientationParameters[5])

        return self.__rotationMatrix

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

    @property
    def isSolved(self):
        """
        True if the exterior orientation is solved

        :return True or False

        :rtype: boolean
        """
        return self.__isSolved

    def ComputeInnerOrientation(self, imagePoints):
        r"""
        Compute inner orientation parameters

        :param imagePoints: coordinates in image space

        :type imagePoints: np.array nx2

        :return: Inner orientation parameters, their accuracies, and the residuals vector

        :rtype: dict

        .. note::

            - Don't forget to update the ``self.__innerOrinetationParameters`` member. You decide the type
            - The fiducial marks are held within the camera attribute of the object, i.e., ``self.camera.fiducialMarks``
            - return values can be a tuple of dictionaries and arrays.

        **Usage example**

        .. code-block:: py

            fMarks = np.array([[113.010, 113.011],
                       [-112.984, -113.004],
                       [-112.984, 113.004],
                       [113.024, -112.999]])
            img_fmarks = np.array([[-7208.01, 7379.35],
                        [7290.91, -7289.28],
                        [-7291.19, -7208.22],
                        [7375.09, 7293.59]])
            cam = Camera(153.42, np.array([0.015, -0.020]), None, None, fMarks)
            img = SingleImage(camera = cam, points = None)
            inner_parameters, accuracies, residuals = img.ComputeInnerOrientation(img_fmarks)
        """
        #  implementing observation vectors
        imagePoints = imagePoints.reshape(np.size(imagePoints), 1)

        fMarks = self.camera.fiducialMarks.reshape(np.size(self.camera.fiducialMarks), 1)

        n = int(len(imagePoints))  # number of observations
        u = 6  # 6 orientation parameters

        A = np.zeros((n, u))  # A matrix (n,u)

        j = 0
        for i in range(len(imagePoints)):
            if i % 2 == 0:
                A[i, 0] = 1;
                A[i, 1] = 0;
                A[i, 2] = fMarks[j];
                A[i, 3] = fMarks[j + 1];
                A[i, 4] = 0
                A[i, 5] = 0
            else:
                A[i, 0] = 0;
                A[i, 1] = 1;
                A[i, 2] = 0;
                A[i, 3] = 0;
                A[i, 4] = fMarks[j];
                A[i, 5] = fMarks[j + 1]
                j += 2

        X = np.dot(la.inv(np.dot(np.transpose(A), A)), np.dot(np.transpose(A), imagePoints))
        v = np.dot(A, X) - imagePoints

        adjustment_results = {"params": X, "residuals": v, "N": np.dot(np.transpose(A), A)}

        self.__innerOrientationParameters = X  # updating the inner orientation params

        return adjustment_results

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
        tx = a0;
        ty = b0
        theta = np.arctan(b1 / b2)
        gamma = np.arctan((a1 * np.sin(theta) + a2 * np.cos(theta)) / (b1 * np.sin(theta) + b2 * np.cos(theta)))
        sx = a1 * np.cos(theta) - a2 * np.sin(theta)
        sy = (a1 * np.sin(theta) + a2 * np.cos(theta)) / np.sin(gamma)

        return {"translationX": tx, "translationY": ty, "rotationAngle": np.rad2deg(theta), "scaleFactorX": sx,
                "scaleFactorY": sy, "shearAngle": np.rad2deg(gamma)}

    def ComputeInverseInnerOrientation(self):
        """
        Computes the parameters of the inverse inner orientation transformation

        :return: parameters of the inverse transformation

        :rtype: dict

        .. warning::

            This function is empty, need implementation

        .. note::

            The inner orientation algebraic parameters are held in ``self.innerOrientationParameters``
            their type is as you decided when implementing
        """
        a0 = self.innerOrientationParameters[0]
        b0 = self.innerOrientationParameters[1]
        a1 = self.innerOrientationParameters[2]
        a2 = self.innerOrientationParameters[3]
        b1 = self.innerOrientationParameters[4]
        b2 = self.innerOrientationParameters[5]

        mat = np.array([[a1[0], a2[0]], [b1[0], b2[0]]])
        mat = la.inv(mat)

        return np.array([a0[0], b0[0], mat[0, 0], mat[0, 1], mat[1, 0], mat[1, 1]]).T

    def CameraToImage(self, cameraPoints):
        """
        Transforms camera points to image points

        :param cameraPoints: camera points

        :type cameraPoints: np.array nx2

        :return: corresponding Image points

        :rtype: np.array nx2


        .. warning::

            This function is empty, need implementation

        .. note::

            The inner orientation parameters required for this function are held in ``self.innerOrientationParameters``

        **Usage example**

        .. code-block:: py

            fMarks = np.array([[113.010, 113.011],
                       [-112.984, -113.004],
                       [-112.984, 113.004],
                       [113.024, -112.999]])
            img_fmarks = np.array([[-7208.01, 7379.35],
                        [7290.91, -7289.28],
                        [-7291.19, -7208.22],
                        [7375.09, 7293.59]])
            cam = Camera(153.42, np.array([0.015, -0.020]), None, None, fMarks)
            img = SingleImage(camera = cam, points = None)
            img.ComputeInnerOrientation(img_fmarks)
            pts_image = img.Camera2Image(fMarks)

        """
        #  setting up the required matrices
        a0 = self.innerOrientationParameters[0]
        b0 = self.innerOrientationParameters[1]
        a1 = self.innerOrientationParameters[2]
        a2 = self.innerOrientationParameters[3]
        b1 = self.innerOrientationParameters[4]
        b2 = self.innerOrientationParameters[5]

        if np.isscalar(a0):

            R = np.array([[a1, a2], [b1, b2]])
            T = np.array([[a0], [b0]])

        else:
            R = np.array([[a1[0], a2[0]], [b1[0], b2[0]]])
            T = np.array([[a0[0]], [b0[0]]])

        cameraPoints = cameraPoints.T
        #  computing the transformation to the image system
        return (T + np.dot(R, cameraPoints)).T

    def ImageToCamera(self, imagePoints):
        """

        Transforms image points to ideal camera points

        :param imagePoints: image points

        :type imagePoints: np.array nx2

        :return: corresponding camera points

        :rtype: np.array nx2

        .. warning::

            This function is empty, need implementation

        .. note::

            The inner orientation parameters required for this function are held in ``self.innerOrientationParameters``


        **Usage example**

        .. code-block:: py

            fMarks = np.array([[113.010, 113.011],
                       [-112.984, -113.004],
                       [-112.984, 113.004],
                       [113.024, -112.999]])
            img_fmarks = np.array([[-7208.01, 7379.35],
                        [7290.91, -7289.28],
                        [-7291.19, -7208.22],
                        [7375.09, 7293.59]])
            cam = Camera(153.42, np.array([0.015, -0.020]), None, None, fMarks)
            img = SingleImage(camera = cam, points = None)
            img.ComputeInnerOrientation(img_fmarks)
            pts_camera = img.Image2Camera(img_fmarks)

        """
        inverse_pars = self.ComputeInverseInnerOrientation()
        imagePoints = imagePoints.T

        if imagePoints.size == 2:
            imagePoints = np.reshape(np.array(imagePoints), (np.size(imagePoints), 1))

        T = np.array([[inverse_pars[0]], [inverse_pars[1]]])
        R = np.array([[inverse_pars[2], inverse_pars[3]], [inverse_pars[4], inverse_pars[5]]])

        return (np.dot(R, imagePoints - T)).T

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


        .. warning::

           - This function is empty, need implementation
           - Decide how the parameters are held, don't forget to update documentation

        .. note::

            - Don't forget to update the ``self.exteriorOrientationParameters`` member (every iteration and at the end).
            - Don't forget to call ``cameraPoints = self.ImageToCamera(imagePoints)`` to correct the coordinates that are sent to ``self.__ComputeApproximateVals(cameraPoints, groundPoints)``
            - return values can be a tuple of dictionaries and arrays.

        **Usage Example**

        .. code-block:: py

            img = SingleImage(camera = cam)
            grdPnts = np.array([[201058.062, 743515.351, 243.987],
                        [201113.400, 743566.374, 252.489],
                        [201112.276, 743599.838, 247.401],
                        [201166.862, 743608.707, 248.259],
                        [201196.752, 743575.451, 247.377]])
            imgPnts3 = np.array([[-98.574, 10.892],
                         [-99.563, -5.458],
                         [-93.286, -10.081],
                         [-99.904, -20.212],
                         [-109.488, -20.183]])
            img.ComputeExteriorOrientation(imgPnts3, grdPnts, 0.3)


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

    def ComputeExteriorOrientation_RzRyRz(self, imagePoints, groundPoints, epsilon):
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


        .. warning::

           - This function is empty, need implementation
           - Decide how the parameters are held, don't forget to update documentation

        .. note::

            - Don't forget to update the ``self.exteriorOrientationParameters`` member (every iteration and at the end).
            - Don't forget to call ``cameraPoints = self.ImageToCamera(imagePoints)`` to correct the coordinates that are sent to ``self.__ComputeApproximateVals(cameraPoints, groundPoints)``
            - return values can be a tuple of dictionaries and arrays.

        **Usage Example**

        .. code-block:: py

            img = SingleImage(camera = cam)
            grdPnts = np.array([[201058.062, 743515.351, 243.987],
                        [201113.400, 743566.374, 252.489],
                        [201112.276, 743599.838, 247.401],
                        [201166.862, 743608.707, 248.259],
                        [201196.752, 743575.451, 247.377]])
            imgPnts3 = np.array([[-98.574, 10.892],
                         [-99.563, -5.458],
                         [-93.286, -10.081],
                         [-99.904, -20.212],
                         [-109.488, -20.183]])
            img.ComputeExteriorOrientation(imgPnts3, grdPnts, 0.3)


        """
        # cameraPoints = self.ImageToCamera(imagePoints)
        cameraPoints = imagePoints
        self.exteriorOrientationParameters[0:3] = np.dot(self.rotationMatrix_RzRyRz,
                                                         self.exteriorOrientationParameters[0:3])
        self.exteriorOrientationParameters = np.add(self.exteriorOrientationParameters,
                                                    np.random.normal(0, 0.01, self.exteriorOrientationParameters.shape))
        l0 = self.__ComputeObservationVector_RzRyRz(groundPoints.T)
        l0 = np.reshape(l0, (-1, 1))
        l = cameraPoints.reshape(np.size(cameraPoints), 1) - l0
        A = self.__ComputeDesignMatrix_RzRyRz(groundPoints.T)

        N = np.dot(A.T, A)
        u = np.dot(A.T, l)
        deltaX = np.dot(la.inv(N), u)

        # update orientation pars
        self.__exteriorOrientationParameters = np.add(self.__exteriorOrientationParameters, np.reshape(deltaX, 6))

        while la.norm(deltaX) > epsilon:
            l0 = self.__ComputeObservationVector_RzRyRz(groundPoints.T)
            l0 = np.reshape(l0, (-1, 1))
            l = cameraPoints.reshape(np.size(cameraPoints), 1) - l0
            A = self.__ComputeDesignMatrix_RzRyRz(groundPoints.T)
            N = np.dot(A.T, A)
            u = np.dot(A.T, l)
            deltaX = np.dot(la.inv(N), u)
            # update orientation pars
            self.__exteriorOrientationParameters = np.add(self.__exteriorOrientationParameters, np.reshape(deltaX, 6))

        # compute residuals
        l_a = np.reshape(self.__ComputeObservationVector_RzRyRz(groundPoints.T), (-1, 1))
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

    def GroundToImage_RzRyRz(self, groundPoints):
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

        R = self.rotationMatrix_RzRyRz
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

            camPoints.append([x, y])

        # return self.CameraToImage(np.array(camPoints))
        return (np.array(camPoints))

    def ImageToRay(self, imagePoints):
        """
        Transforms Image point to a Ray in world system

        :param imagePoints: coordinates of an image point

        :type imagePoints: np.array nx2

        :return: Ray direction in world system

        :rtype: np.array nx3

        .. warning::

           This function is empty, need implementation

        .. note::

            The exterior orientation parameters needed here are called by ``self.exteriorOrientationParameters``
        """
        pass  # delete after implementations

    def ImageToGround_GivenZ(self, imagePoints, Z_values):
        """
        Compute corresponding ground point given the height in world system

        :param imagePoints: points in image space
        :param Z_values: height of the ground points


        :type Z_values: np.array nx1
        :type imagePoints: np.array nx2
        :type eop: np.ndarray 6x1

        :return: corresponding ground points

        :rtype: np.ndarray

        .. warning::

             This function is empty, need implementation

        .. note::

            - The exterior orientation parameters needed here are called by ``self.exteriorOrientationParameters``
            - The focal length can be called by ``self.camera.focalLength``

        **Usage Example**

        .. code-block:: py


            imgPnt = np.array([-50., -33.])
            img.ImageToGround_GivenZ(imgPnt, 115.)

        """
        cameraPoints = self.ImageToCamera(imagePoints)
        cameraPoints = cameraPoints.T
        pars = self.exteriorOrientationParameters
        X0 = pars[0]
        Y0 = pars[1]
        Z0 = pars[2]

        T = np.array([[X0], [Y0], [Z0]])

        omega = pars[3]
        phi = pars[4]
        kappa = pars[5]
        R = Compute3DRotationMatrix(omega, phi, kappa)

        f = self.camera.focalLength

        # allocating memory for return array
        groundPoints = []

        for i in range(len(cameraPoints[1])):
            camVec = np.insert(cameraPoints[:, i], np.size(cameraPoints), -f)
            lam = (Z_values - Z0) / (np.dot(R[2, :], camVec))

            X = X0 + lam * np.dot(R[0, :], camVec)
            Y = Y0 + lam * np.dot(R[1, :], camVec)

            xy = [X, Y, Z_values]
            groundPoints.append(xy)

        groundPoints = np.array(groundPoints)

        return groundPoints

    def castSize(self, scale):
        """
        calculates area of the footprint on the ground
        focalLength and sensorsize in mm
        :param z: diffrent hight from Z (for example at top of the square) (m)
        :return: area in mm2 of FOV footprint
        """
        return self.camera.sensorSize * scale

    def GeneratePointsImg(self, n, ppa):
        """
        Generating grid of points biased by ppa (principal point delta)
        :param n: number of points for each axis
        :param ppa:
        :return:
        """
        x = np.linspace(0, self.camera.sensorSize, n) + ppa[0]
        y = np.linspace(0, self.camera.sensorSize, n) + ppa[1]

        return np.meshgrid(x, y)

    def LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-12):
        """
        Calculates ray intersection point with plane

        :param planePoint:
        :param rayDirection:
        :param rayPoint:
        :param epsilon:
        :return: intersect point on plane
        :rtype: np.ndarray (1x3)
        """

        ndotu = planeNormal.dot(rayDirection)
        if abs(ndotu) < epsilon:
            raise RuntimeError("no intersection or line is within plane")

        w = rayPoint - planePoint
        si = -planeNormal.dot(w) / ndotu
        Psi = w + si * rayDirection + planePoint
        return Psi

    def cameraCorrectionDLT(self, cameraPoints):
        """
        correct camera points to principal point and adding focal length as 'z' value
        :param cameraPoints:  ndarray nx2
        :return: corrected points in camera system 3D
        """
        pts2correct = np.reshape(cameraPoints, (cameraPoints.shape[1], cameraPoints.shape[0]))
        pts2correct = np.vstack(pts2correct, np.ones(1, pts2correct))
        return np.dot(la.inv(self.camera.calibrationMatrix), pts2correct)

    # --------------------- 3D reconstruction -------------------------------------

    def findVanishingPoint(self, point_lines):
        """
        find vanishing point given 2 points on line1 and 2 points on a parallel line2
        :param points_line1: nd.array nx3 (starting points)
        :param point_line2: nd.array nx3 (matching end points)
        :return: vp
        """
        lns = []

        for i in range(0, len(point_lines), 2):
            lns.append(self.__computeLineNormal(point_lines[i][None, :], point_lines[i + 1][None, :]))

        # ln1 = self.__computeLineNormal(point_line1[0, None], point_line1[1, None])
        # ln2 = self.__computeLineNormal(point_line2[0, None], point_line2[1, None])
        # ln3 = self.__computeLineNormal(point_line3[0, None], point_line3[1, None])

        # ln1 = ln1 / ln1[:, -1]
        # ln2 = ln2 / ln2[:, -1]
        # ln3 = ln3 / ln3[:, -1]

        lns = np.vstack(lns)
        # lns = lns / lns[:, -1]

        # A = np.vstack((ln1, ln2, ln3))
        A = lns
        vp = la.solve(np.dot(A[:, 0:2].T, A[:, 0:2]), np.dot(A[:, 0:2].T, -A[:, -1]))

        return np.vstack((vp[:, None], 1)).T

    # def compute_CalibrationMatrix(self, v1, v2, v3):
    #     """
    #     compute calibration parameters given 3 vanishing points of image
    #     :param v1: vanishing point 1 nd.array 1x3
    #     :param v2: vanishing point 2 nd.array 1x3
    #     :param v3: vanishing point 3 nd.array 1x3
    #     :return: calibration matrix K, nd.array 3x3
    #     """
    #     # check if the points are normalized
    #     # if np.all(la.norm(v1, axis=1)) != 1 or np.all(la.norm(v2, axis=1)) != 1 or np.all(la.norm(v3, axis=1)) != 1:
    #     #     v1 = v1 / np.linalg.norm(v1, axis=1)
    #     #     v2 = v2 / np.linalg.norm(v2, axis=1)
    #     #     v3 = v3 / np.linalg.norm(v3, axis=1)
    #
    #     # solving 2 equations with 2 variables - xp, yp
    #     A = np.array([[v2[:, 0] - v3[:, 0], v2[:, 1] - v3[:, 1]], [v2[:, 0] - v1[:, 0], v2[:, 1] - v1[:, 1]]]).astype(
    #         float)[:None, :None, -1]
    #     # b = np.array([[np.dot(v1[:, 0], v2[:, 0]) - np.dot(v1[:, 0], v3[:, 0]) + np.dot(v1[:, 1], v2[:, 1]) - np.dot(
    #     #     v1[:, 1], v3[:, 1])],
    #     #               [np.dot(v3[:, 0], v2[:, 0]) - np.dot(v3[:, 0], v1[:, 0]) + np.dot(v3[:, 1], v2[:, 1]) - np.dot(
    #     #                   v3[:, 1], v1[:, 1])]]).astype(float)
    #     b = np.array([[v1[:, 0] * v2[:, 0] - v1[:, 0] * v3[:, 0] + v1[:, 1] * v2[:, 1] - v1[:, 1] * v3[:, 1]],
    #                   [v3[:, 0] * v2[:, 0] - v3[:, 0] * v1[:, 0] + v3[:, 1] * v2[:, 1] - v3[:, 1] * v1[:, 1]]]).astype(float)[:None, :None, -1]
    #     ans = la.solve(A, b)
    #     xp, yp = ans[0], ans[1]

    def faceNormal_imageSpace(self, r1, r2):
        """
        Compute face normal in image space

        :param r1: vector in the first direction
        :param r2: vector in the second direction

        :type r1: np.array 1x3
        :type r2: np.array 1x3

        :return: face normal
        :rtype: np.array 1x3
        """
        # check if the directions are normalized
        if la.norm(r1) != 1 or la.norm(r2) != 1:
            r1 = r1 / la.norm(r1)
            r2 = r2 / la.norm(r2)

        nl = np.cross(r1, r2)

        return nl

    def faceNormal_objectSpace(self, n):
        """
        Compute the normal in object space

        :param n: normal in image space

        :type n: np.array 1x3

        :return: normal in object space
        :rtype: np.array 1x3
        """
        return np.dot(self.rotationMatrix.T, n)

    def scale_firstFace(self, normal, s, x1, x2):
        """
        Compute the distance of the first normal

        :param normal: face normal in object space
        :param s: known measure in object space
        :param x1: point in image space
        :param x2: point in image space

        :type s: float
        :type x1: np.array 1x3
        :type x2: np.array 1x3

        :return: distance of the first plane from the perspective center in object space

        :rtype: float
        """
        # normalizing image space points
        x1[-1] = -self.camera.focalLength
        x2[-1] = -self.camera.focalLength

        x1 = np.dot(self.rotationMatrix.T, x1) / la.norm(np.dot(self.rotationMatrix, x1))
        x2 = np.dot(self.rotationMatrix.T, x2) / la.norm(np.dot(self.rotationMatrix, x2))

        # getting direction of x1-x2 in object space
        d_hat = (x1 - x2)
        d_hat = d_hat / la.norm(d_hat)

        # x1_hat = np.dot(self.rotationMatrix.T, x1.T)
        # x2_hat = np.dot(self.rotationMatrix.T, x2.T)

        A = np.vstack((x1, -x2))[:, 0:2]
        b = s * d_hat[0:2]
        ans = la.solve(A, b)

        # roh = np.dot(normal.T, x1_hat) * float(ans[0])

        l1 = np.dot(normal, ans[0] * x1)
        l2 = np.dot(normal, ans[1] * x2)

        l = np.mean((l1, l2))

        return l

    def mapPoints(self, normal, distances, points):
        """
        :param normal: face normal object space
        :param distances:  face distance in object space
        :param points: homogeneous coordinates of points in image space

        :type normal: np.array 1x3
        :type distances: float
        :type points: np.array nx3

        :return: points in object space, scale for the next face

        :rtype: tuple

        """

        # Map points to object space
        points_normalized = points.T / la.norm(points, axis=1)  # normalize points in image space
        distances = distances / np.dot(points, normal.T)  # 1D array
        points_objectSpace = (distances * points_normalized.T)  # multiplication of 1D array and a matrix

        # Compute a known measure in the second face
        s = la.norm(points_objectSpace[0] - points_objectSpace[1])

        return points_objectSpace, s

    def rotationMatrix_vanishingPoints(self, vanishingPoint1, vanishingPoint2):
        """
        Compute rotation matrix according to three computed vanishing points that define three cardinal directions

        :param vanishingPoint1: vanishing point
        :param vanishingPoint2: vanishing point

        :type vanishingPoint1: np.array 1x3
        :type vanishingPoint2: np.array 1x3

        :return: rotation matrix
        :rtype: np.array 3x3
        """
        r1 = vanishingPoint1 / la.norm(vanishingPoint1)

        r2 = np.dot(np.eye(max(r1.shape)) - np.dot(r1.T, r1), vanishingPoint2.T)
        r2 = r2 / la.norm(r2)

        R = np.vstack((r1, r2.T, np.cross(r1, r2.T)))

        self.rotationMatrix = R

        return R

    # ---------------------- Private methods ----------------------

    def __computeLineNormal(self, points1, points2):
        """
        Compute the normal of interpretation plane defined by two points and the perspective center

        :param points1: homogeneous coordinates of the first points set
        :param points2: homogeneous coordinates of the second points set

        :type points1: np.array nx3
        :type points2: np.array nx3

        :return: normal of the interpretation plane defined by the two points

        :rtype: np.array nx3

        """

        # check if the points are normalized
        # if np.all(la.norm(points1, axis=1)) != 1 or np.all(la.norm(points2, axis=1)) != 1:
        points1 = points1 / np.linalg.norm(points1, axis=1)
        points2 = points2 / np.linalg.norm(points2, axis=1)

        # Compute the normal of the interpretation plane
        nl = np.cross(points1, points2)

        return nl

    def __ComputeApproximateVals(self, cameraPoints, groundPoints):
        """
        Compute exterior orientation approximate values via 2-D conform transformation

        :param cameraPoints: points in image space (x y)
        :param groundPoints: corresponding points in world system (X, Y, Z)

        :type cameraPoints: np.ndarray [nx2]
        :type groundPoints: np.ndarray [nx3]

        :return: Approximate values of exterior orientation parameters
        :rtype: np.ndarray or dict

        .. note::

            - ImagePoints should be transformed to ideal camera using ``self.ImageToCamera(imagePoints)``. See code below
            - The focal length is stored in ``self.camera.focalLength``
            - Don't forget to update ``self.exteriorOrientationParameters`` in the order defined within the property
            - return values can be a tuple of dictionaries and arrays.

        .. warning::

           - This function is empty, need implementation
           - Decide how the exterior parameters are held, don't forget to update documentation
        """

        # Find approximate values
        cameraPoints = cameraPoints.reshape(np.size(cameraPoints), 1)
        groundPointsXY = groundPoints[0:2, :].T
        groundPointsXY = groundPointsXY.reshape(np.size(groundPointsXY), 1)
        groundPointsZ = groundPoints[2, :].T

        n = int(len(cameraPoints))  # number of observations
        u = 4  # 4 conform parameters

        A = np.zeros((n, u))  # A matrix (n,u)

        j = 0
        for i in range(len(cameraPoints)):
            if i % 2 == 0:
                A[i, 0] = 1
                A[i, 1] = 0
                A[i, 2] = cameraPoints[j]
                A[i, 3] = cameraPoints[j + 1]
            else:
                A[i, 0] = 0
                A[i, 1] = 1
                A[i, 2] = cameraPoints[j + 1]
                A[i, 3] = -cameraPoints[j]
                j += 2

        X = np.dot(la.inv(np.dot(np.transpose(A), A)), np.dot(np.transpose(A), groundPointsXY))

        #  now we can compute the rest of the params
        X0 = X[0]
        Y0 = X[1]
        kappa = np.arctan2(-X[3], X[2])
        lam = np.sqrt(X[2] ** 2 + X[3] ** 2)
        Z0 = np.average(groundPointsZ) + (lam) * self.camera.focalLength

        adjustment_results = {"X0": X0[0], "Y0": Y0[0], "Z0": Z0[0], "omega": 0, "phi": 0,
                              "kappa": np.rad2deg(kappa[0])}

        self.__exteriorOrientationParameters = np.array(
            [X0[0], Y0[0], Z0[0], 0, 0, kappa[0]]).T  # updating the exterior orientation params
        # self.__exteriorOrientationParameters = np.array([202225, 742447, 657.81, 0, 0, kappa[0]]).T
        # return adjustment_results

    def __ComputeApproximateVals_RzRyRz(self, cameraPoints, groundPoints):
        """
        Compute exterior orientation approximate values via 2-D conform transformation

        :param cameraPoints: points in image space (x y)
        :param groundPoints: corresponding points in world system (X, Y, Z)

        :type cameraPoints: np.ndarray [nx2]
        :type groundPoints: np.ndarray [nx3]

        :return: Approximate values of exterior orientation parameters
        :rtype: np.ndarray or dict

        .. note::

            - ImagePoints should be transformed to ideal camera using ``self.ImageToCamera(imagePoints)``. See code below
            - The focal length is stored in ``self.camera.focalLength``
            - Don't forget to update ``self.exteriorOrientationParameters`` in the order defined within the property
            - return values can be a tuple of dictionaries and arrays.
        """

        # Find approximate values
        cameraPoints = cameraPoints.reshape(np.size(cameraPoints), 1)
        groundPointsXY = groundPoints[0:2, :].T
        groundPointsXY = groundPointsXY.reshape(np.size(groundPointsXY), 1)
        groundPointsZ = groundPoints[2, :].T

        n = int(len(cameraPoints))  # number of observations
        u = 4  # 4 conform parameters

        A = np.zeros((n, u))  # A matrix (n,u)

        j = 0
        for i in range(len(cameraPoints)):
            if i % 2 == 0:
                A[i, 0] = 1
                A[i, 1] = 0
                A[i, 2] = cameraPoints[j]
                A[i, 3] = cameraPoints[j + 1]
            else:
                A[i, 0] = 0
                A[i, 1] = 1
                A[i, 2] = cameraPoints[j + 1]
                A[i, 3] = -cameraPoints[j]
                j += 2

        X = np.dot(la.inv(np.dot(np.transpose(A), A)), np.dot(np.transpose(A), groundPointsXY))

        #  now we can compute the rest of the params
        X0 = X[0]
        Y0 = X[1]
        kappa = np.arctan2(-X[3], X[2])
        lam = np.sqrt(X[2] ** 2 + X[3] ** 2)
        Z0 = np.average(groundPointsZ) + (lam) * self.camera.focalLength

        adjustment_results = {"X0": X0[0], "Y0": Y0[0], "Z0": Z0[0], "omega": 0, "phi": 0,
                              "kappa": np.rad2deg(kappa[0])}

        self.__exteriorOrientationParameters = np.array(
            [X0[0], Y0[0], Z0[0], 0.2, 0.2, kappa[0]]).T  # updating the exterior orientation params
        # self.__exteriorOrientationParameters = np.array([202225, 742447, 657.81, 0, 0, kappa[0]]).T
        # return adjustment_results

    def __ComputeObservationVector(self, groundPoints):
        """
        Compute observation vector for solving the exterior orientation parameters of a single image
        based on their approximate values

        :param groundPoints: Ground coordinates of the control points

        :type groundPoints: np.array nx3

        :return: Vector l0

        :rtype: np.array nx1
        """

        n = groundPoints.shape[0]  # number of points

        # Coordinates subtraction
        dX = groundPoints[:, 0] - self.exteriorOrientationParameters[0]
        dY = groundPoints[:, 1] - self.exteriorOrientationParameters[1]
        dZ = groundPoints[:, 2] - self.exteriorOrientationParameters[2]
        dXYZ = np.vstack([dX, dY, dZ])
        rotated_XYZ = np.dot(self.rotationMatrix.T, dXYZ).T

        l0 = np.empty(n * 2)

        # Computation of the observation vector based on approximate exterior orientation parameters:
        l0[::2] = -self.camera.focalLength * rotated_XYZ[:, 0] / rotated_XYZ[:, 2]
        l0[1::2] = -self.camera.focalLength * rotated_XYZ[:, 1] / rotated_XYZ[:, 2]

        return l0

    def __ComputeObservationVector_RzRyRz(self, groundPoints):
        """
        Compute observation vector for solving the exterior orientation parameters of a single image
        based on their approximate values

        :param groundPoints: Ground coordinates of the control points

        :type groundPoints: np.array nx3

        :return: Vector l0

        :rtype: np.array nx1
        """

        n = groundPoints.shape[0]  # number of points

        # Coordinates subtraction
        dX = groundPoints[:, 0] - self.exteriorOrientationParameters[0]
        dY = groundPoints[:, 1] - self.exteriorOrientationParameters[1]
        dZ = groundPoints[:, 2] - self.exteriorOrientationParameters[2]
        dXYZ = np.vstack([dX, dY, dZ])
        rotated_XYZ = np.dot(self.rotationMatrix_RzRyRz.T, dXYZ).T

        l0 = np.empty(n * 2)

        # Computation of the observation vector based on approximate exterior orientation parameters:
        l0[::2] = -self.camera.focalLength * rotated_XYZ[:, 0] / rotated_XYZ[:, 2]
        l0[1::2] = -self.camera.focalLength * rotated_XYZ[:, 1] / rotated_XYZ[:, 2]

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

    def __ComputeDesignMatrix_RzRyRz(self, groundPoints):
        """
            Compute the derivatives of the collinear law (design matrix)

            :param groundPoints: Ground coordinates of the control points

            :type groundPoints: np.array nx3

            :return: The design matrix

            :rtype: np.array nx6

        """
        # initialization for readability
        azimuth = self.exteriorOrientationParameters[3]
        phi = self.exteriorOrientationParameters[4]
        kappa = self.exteriorOrientationParameters[5]

        # Coordinates subtraction
        dX = groundPoints[:, 0] - self.exteriorOrientationParameters[0]
        dY = groundPoints[:, 1] - self.exteriorOrientationParameters[1]
        dZ = groundPoints[:, 2] - self.exteriorOrientationParameters[2]
        dXYZ = np.vstack([dX, dY, dZ])

        rotationMatrixT = self.rotationMatrix_RzRyRz.T
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

        dRTdOmega = Compute3DRotationDerivativeMatrix_RzRyRz(azimuth, phi, kappa, 'azimuth').T
        dRTdPhi = Compute3DRotationDerivativeMatrix_RzRyRz(azimuth, phi, kappa, 'phi').T
        dRTdKappa = Compute3DRotationDerivativeMatrix_RzRyRz(azimuth, phi, kappa, 'kappa').T

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

    def circumcenterTri(self, p1, p2, p3):
        """

        :param p1:
        :param p2:
        :param p3:
        :return:
        """
        ax, bx, cx = p1[0], p2[0], p3[0]
        ay, by, cy = p1[1], p2[1], p3[1]
        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
        uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
        return np.array([ux, uy])


if __name__ == '__main__':

    # sampled_points = Reader.ReadSampleFile(r"Lab7\4points_1vp_4points_2vp.json")
    sampled_points2 = Reader.ReadSampleFile(r"Lab7\new_sampled_points.json")
    another_one = Reader.ReadSampleFile(r"Lab7\something.json")
    borowitz = Reader.ReadSampleFile(r"Lab7\Borowitz.json")
    one_more1 = Reader.ReadSampleFile(r"Lab7\x_axis.json")
    one_more2 = Reader.ReadSampleFile(r"Lab7\y_axis.json")
    one_more3 = Reader.ReadSampleFile(r"Lab7\z_axis.json")

    # points1 = np.hstack(
    #     (sampled_points[0:5, :], np.ones((int(sampled_points[0:5, :].shape[0]), 1))))
    # points2 = np.hstack(
    #     (sampled_points[5:None, :], np.ones((int(sampled_points[5:None, :].shape[0]), 1))))

    # points1 = np.hstack(
    #     #     (sampled_points[::2], np.ones((int(sampled_points[::2].shape[0]), 1))))
    #     # points2 = np.hstack(
    #     #     (sampled_points[1::2], np.ones((int(sampled_points[1::2].shape[0]), 1))))

    # fixing double sampled points
    # sampled_points2[4, :] = sampled_points2[1, :]
    # sampled_points2[6, :] = sampled_points2[3, :]
    # sampled_points2[8, :] = sampled_points2[4, :]
    # sampled_points2[9, :] = sampled_points2[5, :]
    # sampled_points2[10, :] = sampled_points2[0, :]
    #
    # another_one[4, :] = another_one[0, :]
    # another_one[8, :] = another_one[0, :]
    # another_one[6, :] = another_one[2, :]
    # another_one[9, :] = another_one[2, :]
    # another_one[10, :] = another_one[5, :]
    # another_one[11, :] = another_one[7, :]
    # another_one = np.hstack((another_one, np.ones((max(another_one.shape), 1))))
    #
    # borowitz[6, :] = borowitz[0, :]
    # borowitz[8, :] = borowitz[2, :]
    # borowitz[10, :] = borowitz[4, :]
    borowitz = np.hstack((borowitz, np.ones((max(borowitz.shape), 1))))
    another_one = np.hstack((another_one, np.ones((max(another_one.shape), 1))))
    one_more1 = np.hstack((one_more1, np.ones((max(one_more1.shape), 1))))
    one_more2 = np.hstack((one_more2, np.ones((max(one_more2.shape), 1))))
    one_more3 = np.hstack((one_more3, np.ones((max(one_more3.shape), 1))))

    # another_one[4, :] = another_one[0, :]
    # another_one[6, :] = another_one[2, :]
    # another_one = np.hstack((another_one, np.ones((max(another_one.shape), 1))))

    # points1 = np.hstack(
    #     (sampled_points[0:4, :], np.ones((int(sampled_points[0:4, :].shape[0]), 1))))
    # points2 = np.hstack(
    #     (sampled_points[4:8, :], np.ones((int(sampled_points[4:8, :].shape[0]), 1))))
    # points3 = np.hstack(
    #     (sampled_points[8:None, :], np.ones((int(sampled_points[8:None, :].shape[0]), 1))))

    # face1 = np.hstack((sampled_points2[0:4, :], np.ones((int(sampled_points2[0:4, :].shape[0]), 1))))
    # face2 = np.hstack((sampled_points2[4:8, :], np.ones((int(sampled_points2[0:4, :].shape[0]), 1))))
    # face3 = np.hstack((sampled_points2[8:None, :], np.ones((int(sampled_points2[0:4, :].shape[0]), 1))))

    cam = Camera(None, None, {'K1': -0.5104e-8, 'K2': 0.1150e-12},
                 {'P1': -0.8776e-7, 'P2': 0.1722e-7}, None, None)
    img = SingleImage(cam)

    vp1 = img.findVanishingPoint(one_more1)
    vp2 = img.findVanishingPoint(one_more2)
    vp3 = img.findVanishingPoint(one_more3)

    # computing calibration matrix K
    K = cam.compute_CalibrationMatrix(np.reshape(vp1, 3), np.reshape(vp2, 3), np.reshape(vp3, 3))

    # computing lambdas
    lam1 = float(cam.focalLength / np.sqrt(
        (vp1[:, 0] - cam.principalPoint[0]) ** 2 + (vp1[:, 1] - cam.principalPoint[1]) ** 2 + cam.focalLength ** 2))
    lam2 = float(cam.focalLength / np.sqrt(
        (vp2[:, 0] - cam.principalPoint[0]) ** 2 + (vp2[:, 1] - cam.principalPoint[1]) ** 2 + cam.focalLength ** 2))
    lam3 = float(cam.focalLength / np.sqrt(
        (vp3[:, 0] - cam.principalPoint[0]) ** 2 + (vp3[:, 1] - cam.principalPoint[1]) ** 2 + cam.focalLength ** 2))

    # computing R matrix from image to object space
    # R_matrix = img.rotationMatrix_vanishingPoints(vp1, vp2)
    R_matrix = (-1 / cam.focalLength) * np.array(
        [[vp1[:, 0] - cam.principalPoint[0], vp2[:, 0] - cam.principalPoint[0], vp3[:, 0] - cam.principalPoint[0]]
            , [vp1[:, 1] - cam.principalPoint[1], vp2[:, 1] - cam.principalPoint[1], vp3[:, 1] - cam.principalPoint[1]]
            , [-cam.focalLength, -cam.focalLength, -cam.focalLength]])
    R_matrix = np.dot(R_matrix, np.diag(np.array([lam1, lam2, lam3]))).astype(float)
    # update object
    img.rotationMatrix = R_matrix

    R2 = np.hstack(((vp1 / la.norm(vp1)).T, (vp2 / la.norm(vp2)).T, (vp3 / la.norm(vp3)).T))

    # check if what we did is correct - >
    e1 = np.dot(np.dot(R_matrix.T, la.inv(K)), vp1.T)
    e1 = e1 / la.norm(e1)

    # try to reconstruct faces
    face1 = Reader.ReadSampleFile(r"Lab7\face1.json")
    face2 = Reader.ReadSampleFile(r"Lab7\face2.json")
    face3 = Reader.ReadSampleFile(r"Lab7\face3.json")
    # fix shared points
    face2[0] = face1[1]
    face2[-1] = face1[-2]
    face3[0] = face1[0]
    face3[1] = face1[1]
    face3[-2] = face2[1]

    face1 = np.hstack((face1, np.full((max(face1.shape), 1), -cam.focalLength)))
    # face1 = np.hstack((face1, np.ones((max(face1.shape), 1))))
    face2 = np.hstack((face2, np.full((max(face2.shape), 1), -cam.focalLength)))
    face3 = np.hstack((face3, np.full((max(face3.shape), 1), -cam.focalLength)))

    # correcting to 'ideal' camera system
    for i in range(len(face1)):
        face1[i] = np.dot(la.inv(K), face1[i])
        face2[i] = np.dot(la.inv(K), face2[i])
        face3[i] = np.dot(la.inv(K), face3[i])

    # fig_orthographic = plt.figure()
    # ax = fig_orthographic.add_subplot(111, projection='3d')
    # ax.plot(face1[:, 0], face1[:, 1], face1[:, 2], marker='o')
    # ax.plot(face2[:, 0], face2[:, 1], face2[:, 2], marker='^')
    # ax.plot(face3[:, 0], face3[:, 1], face3[:, 2], marker='*')
    # plt.show()

    # computing vanishing points of faces
    vp1face1 = np.cross(face1[0], face2[0])
    vp1face1 = vp1face1 / vp1face1[-1]
    vp2face1 = np.cross(face1[0], face1[-1])
    vp2face1 = vp2face1 / vp2face1[-1]

    vp1face2 = np.cross(face1[1], face1[2])
    vp1face2 = vp1face2 / vp1face2[-1]
    vp2face2 = np.cross(face1[1], face2[1])
    vp2face2 = vp2face2 / vp2face2[-1]

    vp1face3 = np.cross(face1[0], face1[1])
    vp1face3 = vp1face3 / vp1face3[-1]
    vp2face3 = np.cross(face1[0], face3[-1])
    vp2face3 = vp2face3 / vp2face3[-1]

    # computing face normals
    normal_face1 = img.faceNormal_imageSpace(vp1face1, vp2face1)
    normal_face1 = img.faceNormal_objectSpace(normal_face1)

    normal_face2 = img.faceNormal_imageSpace(vp1face2, vp2face2)
    normal_face2 = img.faceNormal_objectSpace(normal_face2)

    normal_face3 = img.faceNormal_imageSpace(vp1face3, vp2face3)
    normal_face3 = img.faceNormal_objectSpace(normal_face3)

    normals = [normal_face1, normal_face2, normal_face3]

    os_points1 = []
    os_points2 = []
    os_points3 = []

    roh1 = img.scale_firstFace(normal_face1, 5, face1[0], face1[1])
    roh2 = img.scale_firstFace(normal_face2, 10, face2[0], face2[1])
    roh3 = img.scale_firstFace(normal_face3, 5, face3[0], face3[1])

    for pnt in face1:
        pnt = np.dot(img.rotationMatrix, pnt)
        li = roh1 / np.dot(normal_face1, pnt)
        os_points1.append(np.dot(li, pnt))

    for pnt in face2:
        pnt = np.dot(img.rotationMatrix, pnt)
        li = roh2 / np.dot(normal_face2, pnt)
        os_points2.append(np.dot(li, pnt))

    for pnt in face3:
        pnt = np.dot(img.rotationMatrix, pnt)
        li = roh3 / np.dot(normal_face3, pnt)
        os_points3.append(np.dot(li, pnt))

    fig_orthographic = plt.figure()
    ax = fig_orthographic.add_subplot(111, projection='3d')
    os_points1 = np.vstack((os_points1, os_points1[0]))
    os_points2 = np.vstack((os_points2, os_points2[0]))
    os_points3 = np.vstack((os_points3, os_points3[0]))
    ax.plot(os_points1[:, 0], os_points1[:, 1], os_points1[:, 2], marker='o')
    # ax.plot(os_points2[:, 0], os_points2[:, 1], os_points2[:, 2], marker='^')
    ax.plot(os_points3[:, 0], os_points3[:, 1], os_points3[:, 2], marker='*')

    plt.show()

    print('hi')
