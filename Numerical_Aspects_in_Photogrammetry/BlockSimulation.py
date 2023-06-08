import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import MatrixMethods as mm
from SingleImage import *
from ImageBlock import *
from Camera import Camera
import copy

# create class for simulate block of images
class SimulateBlock:
    def __init__(self, focal_length, image_size, overlap=0.6 , num_images=2, num_strips=2, tie_pattern='3 mid frame', control_pattern='random block', num_control_points=4 , rotations_sigma=5, altitude=100):
        
        self.focal_length = focal_length # in mm
        self.image_size = image_size # tuple of (width, height) in mm
        self.overlap = overlap # in fraction of image size
        self.num_images = num_images # number of images in every strip
        self.num_strips = num_strips # number of strips in the block
        self.tie_pattern = tie_pattern # '3 mid frame', '4 corners'
        self.control_pattern = control_pattern # 'random block', 'random first image', '5 points'
        self.num_control_points = num_control_points
        self.rotations_sigma = rotations_sigma # in arcsec
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
        rotation = self.simulate_rotation(self.rotations_sigma)
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
                # get block boundaries of the overlapping area
                block_boundaries = self.block_boundaries
                xmin = block_boundaries['xmin'] + (1-self.overlap)*self.image_width/self.scale
                xmax = block_boundaries['xmax'] - (1-self.overlap)*self.image_width/self.scale
                ymin = block_boundaries['ymin'] + (1-self.overlap)*self.image_height/self.scale
                ymax = block_boundaries['ymax'] - (1-self.overlap)*self.image_height/self.scale
                boundaries = {'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax}                                
                control_points = self.generate_random_control_points(boundaries, self.num_control_points)
                
            elif self.control_pattern == 'random first image':
                # get first image boundaries without the overlapping area
                first_image_boundaries = self.images[0].image_ground_bounds
                xmin = first_image_boundaries['xmin']
                xmax = first_image_boundaries['xmax'] - self.overlap*self.image_width/self.scale
                ymin = first_image_boundaries['ymin']
                ymax = first_image_boundaries['ymax'] - self.overlap*self.image_height/self.scale
                boundaries = {'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax}                
                control_points = self.generate_random_control_points(boundaries, self.num_control_points)
                
            elif self.control_pattern == '5 points':
                # points pattern is 4 corners of the block one in the center
                backoff= 5 # m
                block_width = self.block_boundaries['xmax'] - self.block_boundaries['xmin']
                block_height = self.block_boundaries['ymax'] - self.block_boundaries['ymin']                
                control_points = [(self.block_boundaries['xmin']+backoff, self.block_boundaries['ymin']+backoff, 'C0'),
                                    (self.block_boundaries['xmin']+backoff, self.block_boundaries['ymax']-backoff, 'C1'),
                                    (self.block_boundaries['xmax']-backoff, self.block_boundaries['ymin']+backoff, 'C2'),
                                    (self.block_boundaries['xmax']-backoff, self.block_boundaries['ymax']-backoff, 'C3'),
                                    (self.block_boundaries['xmax']-block_width/2, self.block_boundaries['ymax']-block_height/2, 'C4')]
                # convert to dataframe
                control_points = pd.DataFrame(control_points, columns=['X', 'Y', 'name'])
                control_points['Z'] = 0
                                    
            else:
                raise ValueError('Invalid control pattern, choose from "random block" or "random first image"')
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
                
            
    def simulate_image_locations(self):
        """
        Simulate image locations
        Arguments:
        overlap: overlap between images in fraction of image size
        num_images: number of images in each strip
        num_strips: number of horizontal image strips
        Returns:
        image_locations: list of image locations
        """
        overlap = self.overlap
        num_images = self.num_images
        num_strips = self.num_strips
        image_width, image_height = [*self.image_size]
        strip_width = image_width * (num_images / self.scale) * (1 - overlap)
        
        image_locations = []
        for strip_index in range(num_strips):            
            for i in range(num_images):
                x_offset = (i * image_width * (1 - overlap)) / self.scale
                y_offset = strip_index * image_height * (1 - overlap) / self.scale
                image_locations.append((x_offset, y_offset, self.altitude))
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
        for i in range(len(image_locations)):

            # simulate image
            image = self.simulate_image(i, self.camera, self.tie_pattern, image_locations[i])
            
            # get ground coordinates of tie points
            ground_points = image.ImageToGround_GivenZ(image.tie_points[['x', 'y']].values.T, np.zeros(image.tie_points.shape[0]))
            image.tie_points[['X', 'Y', 'Z']] = ground_points.T
            self.tie_points = pd.concat([self.tie_points, image.tie_points])
            self.images.append(image)
        
        # simulate control points
        self.simulate_control_points()
        
        # checking which points are observed in which images
        self.tie_points['num_images'] = 0 # initialize number of images that the point is observed in
        for img in self.images:
            # checking tie points
            # df = self.tie_points.copy()
            self.tie_points['is_point_in_image'] = self.tie_points.apply(lambda row: img.is_point_in_image((row['X'], row['Y'], row['Z'])), axis=1)
            
            # propagate the number of images that the point is observed in
            self.tie_points['num_images'] += self.tie_points['is_point_in_image']
            
             # add the points that are observed in the image to the image.tie_points 
            img.tie_points = self.tie_points[self.tie_points['is_point_in_image']==True].drop(columns=['is_point_in_image']).reset_index(drop=True)
            img.tie_points[['x', 'y']] = img.GroundToImage_fast(img.tie_points[['X', 'Y', 'Z']].values)
            
            # checking control points
            self.control_points['is_point_in_image'] = self.control_points.apply(lambda row: img.is_point_in_image((row['X'], row['Y'], row['Z'])), axis=1)
            
            # add the points that are observed in the image to the image.control_points
            img.control_points = self.control_points[self.control_points['is_point_in_image']==True].drop(columns=['is_point_in_image']).reset_index(drop=True)
            # convert control points to image coordinates
            img.control_points[['x', 'y']] = img.GroundToImage_fast(img.control_points[['X', 'Y', 'Z']].values)
        
        # drop 'is_point_in_image' column
        self.tie_points = self.tie_points.drop(columns=['is_point_in_image'])
        self.control_points = self.control_points.drop(columns=['is_point_in_image'])
        
        # keep only tie points that are observed in more than one image
        
        # in each image keep only the tie points that are observed in more than one image
        for img in self.images:
            # keep only the tie points that are in self.tie_points
            img.tie_points['num_images'] = img.tie_points.apply(lambda row: self.tie_points[(self.tie_points['name'] == row['name']) & (self.tie_points['image_id']==row['image_id'])].num_images.values[0], axis=1)
            img.tie_points = img.tie_points[img.tie_points['num_images']>1].reset_index(drop=True)
            
        self.tie_points = self.tie_points[self.tie_points['num_images']>1].reset_index(drop=True)
        
        
        # insert the index of the tie points in the block to the image tie points
        for i,img in enumerate(self.images):            
            img.tie_points['tie_block_id'] = img.tie_points.apply(lambda row: self.tie_points[(self.tie_points['name'] == row['name']) & (self.tie_points['image_id']==row['image_id'])].index.values[0], axis=1)
            img.tie_points.sort_values(by=['tie_block_id'], inplace=True)
            
                    
        # create block
        block = ImageBlock(copy.deepcopy(self.images), self.tie_points.copy(), self.control_points.copy())
        
        return block
    
    @staticmethod
    def simulate_rotation(sigma):
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
    

    
    @staticmethod
    def add_noise_to_block(block, sigma_rotation=None, sigma_location=None, sigma_image_points=None, sigma_tie_points=None):
        """Add noise to the EOP of the images
        inputs:
        sigma_rotation: standard deviation of rotation angles in arcsec
        sigma_location: standard deviation of location in m
        sigma_tie_points: standard deviation of tie points in microns
        sigma_tie_points: standard deviation of tie points in m
        outputs:
        None
        """
        
        for img in block.images:
            if sigma_rotation is not None:                
                # add noise to rotation angles
                img.exteriorOrientationParameters[3:] += SimulateBlock.simulate_rotation(sigma_rotation)
            if sigma_location is not None:  
                # add noise to location
                img.exteriorOrientationParameters[:3] += np.random.normal(0, sigma_location, 3)

        # add noise to tie points
        block.tie_points[['X', 'Y', 'Z']] += np.random.normal(0, sigma_tie_points, block.tie_points[['X', 'Y', 'Z']].shape)
        # add noise to image samples
        block.tie_points[['x', 'y']] += np.random.normal(0, sigma_image_points, block.tie_points[['x', 'y']].shape) * 1e-3 # convert to mm 
        if 'x' in block.control_points.columns:
            block.control_points[['x', 'y']] += np.random.normal(0, sigma_image_points, block.control_points[['x', 'y']].shape) * 1e-3 # convert to mm
        
        
    