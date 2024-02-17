'''
Function:     white_image


Arguments:    x_pixels - Number of pixels in the horizontal (azimuthal) direction.
              y_pixels - Number of pixels in the vertical (elevational) direction.
              upsample - The degree of upsampling present in the final image.
              
           
Output:       image - A binary spherical image with size (x_pixels,y_pixels) consisting of size
                      (360/x_pixels*upsample,180/y_pixels*upsample) degree blocks. Each block has all pixels set to 0
                      or 1, randomly and with equal probability.
           
           
Description:  Generates a binary, spherical image consisting of x_pixels horizontal (azimuthal) pixels, and y_pixels
              vertical (elevational) pixels. First, a white, binary image is generated at the resolution
              (x_pixels/upsample,y_pixels/upsample), and then upsampled to the desired resolution.


Authors:      James Trousdale - jamest212@gmail.com
'''


import numpy as np

#                                              #
#                                              #
###### definition of function white_image ######
#                                              #
#                                              #

def white_image(x_pixels,y_pixels,upsample):
    # Generate a white, binary image at the coarser resolution
    image_0 = (np.random.rand(y_pixels/upsample,x_pixels/upsample)<0.5)
    
    image = np.zeros((y_pixels,x_pixels))
    
    # Upsample to obtain a binary image where blocks of upsample^2 pixels consist either of all 0's or all 1's.
    for i in xrange(y_pixels):
        for j in xrange(x_pixels):
            image[i,j] = image_0[i/upsample,j/upsample]
            
    return image
    
#                                                  #
#                                                  #
###### default code executed when white_image ###### 
###### is called without parameters           ######
#                                                  #
#                                                  #  
    
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    x_pixels = 360
    y_pixels = 180
    upsample = 4
    
    image = white_image(x_pixels,y_pixels,upsample)
    
    plt.imshow(image,cmap='gray')
    plt.show()