import os, fnmatch, code
import cv2
from torchvision import transforms
import matplotlib.pyplot as plt


def IoU(path, data_num, threshold):
	heatmap = cv2.imread(path+'/heatmap_mask_{}_{}.jpg'.format(threshold, data_num))
	mask = cv2.imread(path+'/mask{}.jpg'.format(data_num))
	smooth = 1e-6
	intersection = (heatmap & mask).sum()
	union = (heatmap | mask).sum()

	iou = (intersection + smooth) / (union + smooth)     # smooth is added to the devision to avoid 0/0

	return iou


#folder = '../BrainTumorClassification/CAM_vgg16'
#folder = '../BrainTumorClassification/CAM_alexnet'
folder = '../BrainTumorClassification/final_result'
threshold = 'Q90'
total_size = len(fnmatch.filter(os.listdir(folder), 'heatmap_mask_{}_*.jpg'.format(threshold)))
total_IoU = 0
for i in range(1, total_size+1):
	iou = IoU(folder, i, threshold)
	#print('IoU is {:.6f}'.format(iou))
	total_IoU += iou

mIoU = total_IoU/total_size
print('mIoU with threshold Q90 is {:.10f}'.format(mIoU))



threshold = 'Q95'
total_size = len(fnmatch.filter(os.listdir(folder), 'heatmap_mask_{}_*.jpg'.format(threshold)))
total_IoU = 0
for i in range(1, total_size+1):
	iou = IoU(folder, i, threshold)
	#print('IoU is {:.6f}'.format(iou))
	total_IoU += iou


mIoU = total_IoU/total_size
print('mIoU with threshold Q95 is {:.10f}'.format(mIoU))




