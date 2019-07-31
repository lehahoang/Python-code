#import matplotlib.pyplot as plt
'''
	@ In this python code, the following are executed:
	@ Load the entire original images of evaluation dataset to a listdir
	@ Preprocess the original images to 224x224x3 RGB images
	@ Save new images to .hkl format files correspondingly

'''
import numpy as np
import scipy.io
import cv2
import os
import hickle as hkl

# Load the data

data_dir= '/nas/lhoang/data/evaluation/'# Path to original images
saved_dir='/nas/lhoang/data/hkl_file/'  # Path to saved hkl files
fn = os.listdir(data_dir) # load file names into a list

fns = [data_dir+ name for name in fn]# concatenate the directory path with the file names
									 # (i.e. /homes/lhoang/.../IMG0001.JPEG)
numpy = True

for i in range(len(fns)):
	if i%2000==0:
		print('%d/%d' % (i,len(fns))) # Print out (2000/500000)
	name = fn[i].replace('.JPEG','')+'.hkl' # Remove the .JPEG extension and concatenate with .hkl extension
											# (i.e., IMG0001.JPEG -> IMG0001.hkl)
	name = saved_dir+name 				     # Concatenate the saved directory with the hkl file
	img = cv2.imread(fns[i])
	#print img.shape
	height, width, channels = img.shape     # Take the size of height, width and channles of an image
	new_height = int((height*256)/min(img.shape[:2])) # Resize the image to 256x256
	new_width  = int((width*256)/min(img.shape[:2]))  # Method is CUBIC
	img_new = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC).astype(np.float32)
	#cropping
	height, width, channels = img_new.shape
	start_height  = int((height-224)/2)
	start_width   = int((width-224)/2)
	img = img_new[start_height:start_height+224,start_width:start_width+224]
	img[:,:,0] -= 103.939
	img[:,:,1] -= 116.779
	img[:,:,2] -= 123.68
	img			= img.transpose((2,0,1))
	# new image has the shape of (224,224,3)
	img = np.expand_dims(img, axis =0) # now the shape of the image is (1,224,224,3)
	#img = np.swapaxes(img, 1,2)
	#img = np.swapaxes(img, 1,3) # swap the axes to new shape (1,3,224,224)
								# This is neccesary to leverage the existing code in
								# Ares simulator. We will take other approaches if needed
	hkl.dump(img, name, mode = 'w' ) # Save the new image to a .hkl name



	# array_image_data = np.expand_dims(array_image_data,axis=0)
	# print ("shape after the expansion:",array_image_data.shape)
	# array_image_data = preprocess_input(array_image_data)
	# print ("Final shape:",array_image_data.shape)
