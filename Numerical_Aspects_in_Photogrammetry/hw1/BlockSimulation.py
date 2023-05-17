import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import MatrixMethods as mm
from SingleImage import *
from ImageBlock import *
from Camera import Camera

# create class for simulate block of images
class SimulateBlock:
    def __init__(self, focal_length, image_size, overlap=0.6 , num_images=2, tie_pattern='3 mid frame', control_pattern='random block', num_control_points=4 , rotaions_sigma=5, altitude=100):
        
        self.focal_length = focal_length # in mm
        self.image_size = image_size # tuple of (width, height) in mm
        self.overlap = overlap # in fraction of image size
        self.num_images = num_images
        self.tie_pattern = tie_pattern # '3 mid frame', '4 corners'
        self.control_pattern = control_pattern # 'random entire block', 'random first image'
        self.num_control_points = num_control_points
        self.rotaions_sigma = rotaions_sigma # in arcsec
        self.altitude = altitude # in m
        self.images = []
        self.tie_points = pd.DataFrame(columns=['x', 'y', 'name', 'image_id', 'X', 'Y', 'Z'])
        self.control_points = []
        self.camera = Camera(focal_length, image_size)
        self.block = []
        
    
    @property
    def image_height(self):
        return self.image_size[1]
    
    @property
    def image_width(self):
        return self.image_size[0]
    
    @property
    def scale(self):
        return self.focal_length/self.altitude
    
    # create property for block boundaries in ground coordinates
    # return dictionary of (xmin, xmax, ymin, ymax)
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
        
        
    
        
    def simulate_image(self, image_id, camera, tie_pattern, location):
        # simulate tie points
        tie_points = self.simulate_tie_points(tie_pattern, image_id=image_id)
        # simulate rotation angles
        rotation = self.simulate_rotation(self.rotaions_sigma)
        exteriorOrientationParameters = np.concatenate((location, rotation))
        # initialize image
        image = SingleImage(camera, image_id, exteriorOrientationParameters, tie_points)
        
        
        return image
    
    def simulate_tie_points(self, tie_pattern, backoff=0.9, image_id=0):
        """Simulate tie points in the image
        inputs: 
        tie_pattern: '3 mid frame', '4 corners'
        backoff: backoff from the edge of the image in fraction of image size
        outputs:
        tie_points: list of tie points in the image
        """
        if tie_pattern == '3 mid frame':
            tie_points = [(0, -self.image_height/2*(backoff), 'T0', image_id),
                          (0, 0, 'T1', image_id),
                          (0, +self.image_height/2*backoff, 'T2', image_id)]
        elif tie_pattern == '4 corners':
            tie_points = [(-self.image_width/2*backoff, -self.image_height/2*backoff, 'T0', image_id),
                          (self.image_width/2*backoff, -self.image_height/2*backoff, 'T1', image_id),
                          (-self.image_width/2*backoff, self.image_height/2*backoff, 'T2', image_id),
                          (self.image_width/2*backoff, self.image_height/2*backoff, 'T3', image_id)]
        else:
            raise ValueError('Invalid tie pattern, choose from "3 mid frame" or "4 corners"')
            
            
        # convert to dataframe
        tie_points = pd.DataFrame(tie_points, columns=['x', 'y', 'name', 'image_id'])
        return tie_points

    
    # simulate control points in ground coordinates
    def simulate_control_points(self):
        if len(self.images)==0:
            print('No images in block')
        else:
            # check control pattern
            if self.control_pattern == 'random block':
                # get block boundaries
                block_boundaries = self.block_boundaries
                # generate random control points
                control_points = self.generate_random_control_points(block_boundaries, self.num_control_points)
            elif self.control_pattern == 'random first image':
                # get first image boundaries
                first_image_boundaries = self.images[0].image_ground_bounds
                # generate random control points
                control_points = self.generate_random_control_points(first_image_boundaries, self.num_control_points)
            else:
                print('Invalid control pattern')
            # add image_id
            self.control_points = control_points
    
    # generate random control points
    # static method
    @staticmethod
    def generate_random_control_points(boundaries, num_control_points):
        """Generate random control points
        inputs:
        boundaries: dictionary of (xmin, xmax, ymin, ymax)
        num_control_points: number of control points
        outputs:
        control_points: list of control points
        """
        # generate random control points
        control_points = []
        for i in range(num_control_points):
            control_points.append((np.random.uniform(boundaries['xmin'], boundaries['xmax']), np.random.uniform(boundaries['ymin'], boundaries['ymax']), 'C'+str(i)))
        # convert to dataframe
        control_points = pd.DataFrame(control_points, columns=['X', 'Y', 'name'])
        control_points['Z'] = 0
        return control_points
                
    def simulate_rotation(self, sigma):
        """Simulate rotation angles in arcsec
        inputs:
        sigma: standard deviation of rotation angles in arcsec
        outputs:
        rotation: rotation angles in arcsec
        """
        # Convert sigma to radians
        sigma_rad = np.radians(sigma / 3600)  # Convert arcseconds to degrees, then to radians

        # Generate random rotation angles with sigma=2 arcseconds
        rotation = np.random.normal(0, sigma_rad, 3)
        return rotation
        
    def simulate_image_locations(self):
        """Simulate image locations
        inputs:
        overlap: overlap between images in fraction of image size
        num_images: number of images
        outputs:
        image_locations: list of image locations
        """
        # calculate image locations
        image_locations = []
        for i in range(self.num_images):
            image_locations.append((i*self.image_width*(1-self.overlap)*(1/self.scale), 0, self.altitude))
        return image_locations

    def simulate_block(self):
        """Simulate block of images
        inputs:
        self
        outputs:
        block: list of images
        """
        # calculate image locations
        image_locations = self.simulate_image_locations()
        
        
        # simulate images
        for i in range(self.num_images):

            # simulate image
            image = self.simulate_image(i, self.camera, self.tie_pattern, image_locations[i])
            
            # get ground coordinates of tie points
            ground_points = image.ImageToGround_GivenZ(image.tie_points[['x', 'y']].values.T, np.zeros(image.tie_points.shape[0]))
            image.tie_points[['X', 'Y', 'Z']] = ground_points.T
            self.tie_points = pd.concat([self.tie_points, image.tie_points])
            self.images.append(image)
        
        # simulate control points
        self.GC_points = self.simulate_control_points()
        
        # checking which points are observed in which images
        self.tie_points['num_images'] = 0 # initialize number of images that the point is observed in
        for img in self.images:
            # checking tie points
            # df = self.tie_points.copy()
            self.tie_points['is_point_in_image'] = self.tie_points.apply(lambda row: img.is_point_in_image((row['X'], row['Y'], row['Z'])), axis=1)
            
            # prpagate the number of images that the point is observed in
            self.tie_points['num_images'] += self.tie_points['is_point_in_image']
            
             # add the points that are observed in the image to the image.tie_points 
            img.tie_points = self.tie_points[self.tie_points['is_point_in_image']==True].drop(columns=['is_point_in_image']).reset_index(drop=True)
            
            # checking control points
            self.control_points['is_point_in_image'] = self.control_points.apply(lambda row: img.is_point_in_image((row['X'], row['Y'], row['Z'])), axis=1)
            
            # add the points that are observed in the image to the image.control_points
            img.control_points = self.control_points[self.control_points['is_point_in_image']==True].drop(columns=['is_point_in_image']).reset_index(drop=True)
        
        # drop 'is_point_in_image' column
        self.tie_points = self.tie_points.drop(columns=['is_point_in_image'])
        self.control_points = self.control_points.drop(columns=['is_point_in_image'])
        
        # keep only tie points that are observed in more than one image
        self.tie_points = self.tie_points[self.tie_points['num_images']>1].reset_index(drop=True)
        
        # insert the index of the tie points in the block to the image tie points
        for img in self.images:
            img.tie_points['tie_block_id'] = img.tie_points.apply(lambda row: self.tie_points[(self.tie_points['name'] == row['name']) & (self.tie_points['image_id']==row['image_id'])].index.values[0], axis=1)
            img.tie_points.sort_values(by=['tie_block_id'], inplace=True)
                    
        # create block
        block = ImageBlock(self.images, self.tie_points, self.control_points)
        return block
    

    