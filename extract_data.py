import os, random
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch

import code

########## Extract mat data from .zip file ##########

if not os.path.exists('BrainData'):
	os.mkdir('BrainData')

# extract .mat dataset .zip file
with zipfile.ZipFile('brain_tumor_data.zip') as item:
	item.extractall('BrainData')   # file contains 4 zip files of brain tumor data

data_folder = 'BrainData'

def Extract_data(path):
	zip_files = [file for file in os.listdir(path) if file not in ['cvind.mat','README.txt']]
	data = {}

	if not os.path.exists('BrainData/dataset'):
		os.mkdir('BrainData/dataset')

	for file_to_unzip in zip_files:
		with zipfile.ZipFile('BrainData/{}'.format(file_to_unzip)) as item:
			item.extractall('BrainData/dataset')


########## Convert image data into image and mask ##########

def convert_images(data_path, image_path, mask_path):
	for data in range(1, 3065):
		file = h5py.File(data_path+'/{}.mat'.format(data), 'r')
		image = np.array(file['cjdata']['image'], dtype=np.float32)         # has size of (512,512)
		plt.imsave(image_path+'/{}.jpg'.format(data), image, cmap='gray')
		mask = np.array(file['cjdata']['tumorMask'], dtype=np.float32)      # has size of (512,512)
		plt.imsave(mask_path+'/{}.jpg'.format(data), mask, cmap='gray')

		#PID = ''.join(chr(c) for c in (file['cjdata']['PID']))


########## Split data into 5 files ##########

def split_class(data_path):
	# total 3064 data available
	# there are three classes (1, 2, 3)
	# split data according to class

	class_1, class_2, class_3 = [], [], []
	for data in range(1, 3065):
		file = h5py.File(data_path+'/{}.mat'.format(data), 'r')
		label = int(file['cjdata']['label'][0][0])
		if label == 1:
			class_1.append(data)
		elif label == 2:
			class_2.append(data)
		elif label == 3:
			class_3.append(data)

	return class_1, class_2, class_3


def create_datafile(data_path):
	# split the dataset into 5 data files   -> [613, 613, 613, 613, 612]
	# there are 708 Meningioma tumor images -> [142, 142, 142, 141, 141]
	# there are 1426 Glioma tumor images    -> [285, 285, 285, 286, 285]
	# there are 930 Pituitary tumor images  -> [186, 186, 186, 186, 186]
	
	class1, class2, class3 = split_class(data_path)
	class1_split_num = [142, 142, 142, 141, 141]
	class2_split_num = [285, 285, 285, 286, 285]
	class3_split_num = [186, 186, 186, 186, 186]

	for file_num in range(5):
		file_out = open('dataset{}.csv'.format(file_num),'w')

		for data1 in class1[:class1_split_num[file_num]]:
			file_out.write(str(data1)+',1\n')
		del class1[:class1_split_num[file_num]]

		for data2 in class2[:class2_split_num[file_num]]:
			file_out.write(str(data2)+',2\n')
		del class2[:class2_split_num[file_num]]

		for data3 in class3[:class3_split_num[file_num]]:
			file_out.write(str(data3)+',3\n')
		del class3[:class3_split_num[file_num]]





if __name__ == "__main__":
	if not os.path.exists('BrainData/image'):
		os.mkdir('BrainData/image')
	if not os.path.exists('BrainData/mask'):
		os.mkdir('BrainData/mask')

	data_path = 'BrainData/dataset'
	image_path = 'BrainData/image'
	mask_path = 'BrainData/mask'

	#Extract_data(data_folder)
	#convert_images(data_path, image_path, mask_path)
	create_datafile(data_path)
