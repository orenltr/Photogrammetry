class Camera:
    def __init__(self, focal_length, sensor_size):
        self.focal_length = focal_length
        self.sensor_size = sensor_size # tuple of (width, height) in mm