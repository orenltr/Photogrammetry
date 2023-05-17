import numpy as np
from matplotlib import pyplot as plt
from SingleImage import SingleImage
import Camera
from ImagePair import ImagePair
from ImageBlock import ImageBlock
import xlsxwriter
import datetime


def dist(p1,p2):
    return np.sqrt((p2[0]-p1[0])**2+(p2[1]-p1[1])**2+(p2[2]-p1[2])**2)

# if __name__ == '__main__':

#     # load data
#     T_data = np.loadtxt('T_samples.txt', dtype=float, skiprows=1)
#     T_data[:, :2] = np.uint8(T_data[:, :2])
#     GC_data = np.loadtxt('G_samples.txt', dtype=float, skiprows=1)
#     GC_data[:, :2] = np.uint8(GC_data[:, :2])
#     GC_coordinates = np.loadtxt('G_coordinates.txt', dtype=float, skiprows=1)
#     estimate_exterior_orientation = np.loadtxt('estimate_values.txt', dtype=float)
#     image_size = 25.4  # 25.4x25.4 [mm]
#     focal_length = 30   # [mm]

#     # removing wrong data
#     GC_data = GC_data[np.abs(GC_data[:, 2]) < image_size/2]
#     GC_data = GC_data[np.abs(GC_data[:, 3]) < image_size/2]

#     # arranging data
#     images = []
#     cam = Camera.Camera(focal_length,np.array([image_size/2,image_size/2]),0,0,0)
#     for i in range(int(max(T_data[:, 0]))):
#         T_samples = T_data[T_data[:, 0] == i+1, 1:]
#         G_samples = GC_data[GC_data[:, 0] == i+1, 1:]
#         im = SingleImage(cam, T_samples, G_samples, estimate_exterior_orientation[i])
#         images.append(im)

#     # creating ImageBlock instance
#     block = ImageBlock(images,GC_coordinates,T_data,GC_data)

#     # calculate tie points estimate values
#     block.T_estimate_values()

#     # export tie points coordinate to excel
#     # workbook = xlsxwriter.Workbook('T_estimate_coordinates.xlsx')
#     # worksheet = workbook.add_worksheet()
#     # for col, data in enumerate(T_estimate_coordinates.T):
#     #     worksheet.write_column(0, col, data)
#     #
#     # workbook.close()
#     # drawing block data
#     block.draw_block()

#     # drawing A matrix
#     A = block.ComputeDesignMatrix()
#     plt.spy(A)
#     plt.grid()
#     plt.show()
#     X,RMSE,S = block.BundleAdjustment(0.01,100)

#     # std results
#     S = np.sqrt(np.diag(S))
#     S_orientation = np.reshape(S[:len(block.images)*4],(9,4)).T

#     # workbook = xlsxwriter.Workbook('sigma.xlsx')
#     # worksheet = workbook.add_worksheet()
#     # for col, data in enumerate(S1.T):
#     #     worksheet.write_column(0, col, data)
#     #
#     # workbook.close()
#     # # # export exteriorOrientationParameters coordinate to excel
#     # # workbook = xlsxwriter.Workbook('exteriorOrientationParameters.xlsx')
#     # # worksheet = workbook.add_worksheet()
#     # # for col, im in enumerate(block.images):
#     # #     a = np.copy(im.exteriorOrientationParameters)
#     # #     a = np.reshape(a,(6,1))
#     # #     worksheet.write_column(0, col, a.T )
#     # # workbook.close()





    
    



