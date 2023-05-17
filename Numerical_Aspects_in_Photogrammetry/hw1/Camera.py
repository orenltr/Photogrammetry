
class Camera(object):

    def __init__(self, focal_length, principal_point, pix_size, camera_parameters):
        """
        Initialize the Camera object

        :param focal_length: focal length of the camera(mm)
        :param principal_point: principle point
        :param pix_size: pixel size in mm
        :param camera_parameters: 10 camera parameters

        :type focal_length: double
        :type principal_point: np.array
        :type pix_size: pixel size in mm
        """
        # private parameters
        self.__focal_length = focal_length
        self.__principal_point = principal_point
        self.__pix_size = pix_size
        self.__camera_parameters = camera_parameters

    @property
    def focalLength(self):
        """
        Focal length of the camera

        :return: focal length

        :rtype: float

        """
        return self.__focal_length

    @focalLength.setter
    def focalLength(self, val):
        """
        Set the focal length value

        :param val: value for setting

        :type: float

        """

        self.__focal_length = val

    @property
    def principalPoint(self):
        """
        Principal point of the camera

        :return: principal point coordinates

        :rtype: np.ndarray

        """

        return self.__principal_point

    @property
    def pixelSize(self):
        """
        pixel size of the camera

        :return: pixel size in mm

        :rtype: float

        """
        return self.__pix_size

    @pixelSize.setter
    def pixelSize(self, val):
        """
        Set the pixel size value

        :param val: value for setting

        :type: float

        """

        self.__pix_size = val

    @property
    def parameters(self):
        """
        10 camera parameters

        :return: 10 camera parameters

        :rtype: dict

        """
        return self.__camera_parameters

    @parameters.setter
    def parameters(self, val):
        """
        Set the camera parameters

        :param val: value for setting

        :type: dict

        """

        self.__camera_parameters = val
