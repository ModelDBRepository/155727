'''
Function:	 natural_image


Arguments:	x_pixels - Number of pixels in the horizontal (azimuthal) direction.
	        y_pixels - Number of pixels in the vertical (elevational) direction.
		image_lib_path - (optional) Path to directory containing a Python pickle file 'image_lib.dat', which holds
				 a dictionary with entry 'image_lib'. The image library should consist of a 3-dimensional
				 array of size (N,302,302), where each slice [i,:,:] contains a DxD downsampling of a
				 natural scene from the van Hateren dataset. Assuming the spherical image will be of
				 resolution 180x360, a value of D = ~300 seems to work well.
			  
		   
Output:	   image - A spherical image with six sections composed of natural scenes.
		   
		   
Description:  Generates a spherical image composed of various, randomly selected natural scenes from the van Hateren
	      dataset. This is done by first randomly selecting six images from a downsampled subcollection of the
	      van Hateren dataset, and pasting them to the surfaces of a cube. This cube is then projected on to the
	      surface of the sphere. Each face is projected via what resembles a partial stereographic projection,
	      except the projection point is the center of the sphere. The six faces are defined in rectangular
	      coordinates as follows:
			  
	      * Face 1: |x|>|y|, |x|>|z
			  
	      For example, the rectangular coordinates of a point lying on face 1 would have |x| > |z|, |x| > |y|. Given
	      that a point lies on face 1, we then use its y,z coordinates to calculate the relative position within
	      that square that the projection lies, and set the pixel on the spherical image equal to the average of
	      the 3x3 square surrounding the projected pixel.


Authors:      James Trousdale - jamest212@gmail.com
'''


__all__ = ["natural_image"]

import numpy as np
import pickle,os

#                                                #
#                                                #
###### definition of function natural_image ######
#                                                #
#                                                #

def natural_image(x_pixels,y_pixels,image_lib_path='.'):
	
	image = np.zeros((y_pixels,x_pixels))
	
	if image_lib_path[-1] == '/':
		image_lib_path = image_lib_path[:-1]
	
	# Import the image library.
	with open(os.path.expanduser(image_lib_path)+'/image_lib.dat',"rb") as fp:
		image_lib = pickle.load(fp)['image_lib']
		
	
	# Select six random images from the subset of the van Hateren dataset which will be applied to the cube faces.
	im1 = np.flipud(image_lib[np.random.randint(np.size(image_lib,0)),:,:])
	im2 = np.flipud(image_lib[np.random.randint(np.size(image_lib,0)),:,:])
	im3 = np.flipud(image_lib[np.random.randint(np.size(image_lib,0)),:,:])
	im4 = np.flipud(image_lib[np.random.randint(np.size(image_lib,0)),:,:])
	im_top = np.flipud(image_lib[np.random.randint(np.size(image_lib,0)),:,:])
	im_bot = np.flipud(image_lib[np.random.randint(np.size(image_lib,0)),:,:])
	
	# Normalize each image to be used so that pixel values range over [0,1].
	im1 = (np.double(im1)-np.min(im1))/(np.max(im1)-np.min(im1))
	im2 = (np.double(im2)-np.min(im2))/(np.max(im2)-np.min(im2))
	im3 = (np.double(im3)-np.min(im3))/(np.max(im3)-np.min(im3))
	im4 = (np.double(im4)-np.min(im4))/(np.max(im4)-np.min(im4))
	im_top = (np.double(im_top)-np.min(im_top))/(np.max(im_top)-np.min(im_top))
	im_bot = (np.double(im_bot)-np.min(im_bot))/(np.max(im_bot)-np.min(im_bot))

	# Determine the resolution of the images pasted on the surface of the sphere. Two is subtracted because we will use 
	# in the spherical images 3x3 averages of pixels in the original images on the surface of the cube.
	im_res = np.size(im1,0)-2 
	
	
	# The vectors theta_vals and phi_vals contain the azimuth and elevation of the pixel centers, respectively.
	theta_vals = np.linspace(0,2*np.pi,np.size(image,1)+1,endpoint=True)
	theta_vals = (theta_vals[:-1] + theta_vals[1:])/2
	phi_vals = np.linspace(0,np.pi,np.size(image,0)+1,endpoint=True)
	phi_vals = (phi_vals[:-1] + phi_vals[1:])/2
	
	# x,y,z contain the rectangular coordinates of the pixel centers
	x = np.outer(np.cos(theta_vals),np.sin(phi_vals))
	y = np.outer(np.sin(theta_vals),np.sin(phi_vals))
	z = np.cos(phi_vals)	
		
	for i in range(y_pixels):
		for j in range(x_pixels):
			# Determine which face this point lies on, and then the corresponding indices of the pixel on the square
			# face which has its center nearest to the projected point. The spherical image pixel is set to an average
			# of the 3x3 block surrounding the projected point on the cube face.
			if(np.abs(x[j,i]) > np.abs(y[j,i]) and np.abs(x[j,i]) > np.abs(z[i])):
				# y,z are first scaled inversely by x which gives the rectangular coordinates of the corresponding point
				# on the cube face (for faces 1,3, the corresponding cube face will have |x| = 1, so the problem is to
				# determine the rectangular coordinates on the cube which lies on the ray emitting from the origin and
				# passing through the coordinate x,y,z on the unit sphere).These coordinates lie within [-1,1], so they
				# are scaled to lie in [0,1], then snapped to a corresponding pixel in the image on the surface of the
				# cube.
				y_ind = np.ceil((y[j,i]/np.abs(x[j,i])+1)/2*im_res)
				z_ind = np.ceil((z[i]/np.abs(x[j,i])+1)/2*im_res)
				if(x[j,i]>0):  # Face 1
					image[i,j] = np.mean(im1[z_ind-1:z_ind+1,y_ind-1:y_ind+1])
				else:          # Face 3
					image[i,j] = np.mean(im3[z_ind-1:z_ind+1,y_ind-1:y_ind+1])
			elif(np.abs(y[j,i]) > np.abs(z[i])):
				x_ind = np.ceil((x[j,i]/np.abs(y[j,i])+1)/2*im_res)
				z_ind = np.ceil((z[i]/np.abs(y[j,i])+1)/2*im_res)
				if(y[j,i]>0):  # Face 2
					image[i,j] = np.mean(im2[z_ind-1:z_ind+1,x_ind-1:x_ind+1])
				else:          # Face 4
					image[i,j] = np.mean(im4[z_ind-1:z_ind+1,x_ind-1:x_ind+1])
			else:
				x_ind = np.ceil((x[j,i]/np.abs(z[i])+1)/2*im_res)
				y_ind = np.ceil((y[j,i]/np.abs(z[i])+1)/2*im_res)
				if(z[i]>0):    # Top face
					image[i,j] = np.mean(im_top[y_ind-1:y_ind+1,x_ind-1:x_ind+1])
				else:          # Bottom face
					image[i,j] = np.mean(im_bot[y_ind-1:y_ind+1,x_ind-1:x_ind+1])
					
	return image


#                                                    #
#                                                    #
###### default code executed when natural_image ###### 
###### is called without parameters             ######
#                                                    #
#                                                    #
    

if __name__ == "__main__":
	import matplotlib.pyplot as plt
	
	x_pixels = 360
	y_pixels = 180
	
	image = natural_image(x_pixels,y_pixels,image_lib_path='.')
	
	plt.imshow(image,cmap='gray')
	plt.show()	