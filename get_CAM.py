import os, code, cv2, PIL
import torch
import numpy as np
from torch.nn import functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models
import imageio
from torch.autograd import Variable
from PIL import Image
from skimage import io, 
from models_vgg import vggNet16, vggNet19
from models_alexnet import CNN
from data_process import Brain_data

def create_CAM(data, model, result_path, i):
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	features_blobs = []
	def hook_feature(module, input, output):
		features_blobs.append(output.cpu().data.numpy())

	model._modules.get('features').register_forward_hook(hook_feature)

	# get the softmax weight
	params = list(model.parameters())
	weight_softmax = np.squeeze(params[-2].data.numpy())

	def returnCAM(feature_conv, weight_softmax, class_idx):
		# generate the class activation maps upsample to 256x256
		size_upsample = (256, 256)
		bz, nc, h, w = feature_conv.shape
		output_cam = []
		for idx in class_idx:
			cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
			cam = cam.reshape(h, w)
			cam = cam - np.min(cam)
			cam_img = cam / np.max(cam)
			cam_img = np.uint8(255 * cam_img)
			output_cam.append(cv2.resize(cam_img, size_upsample))
		return output_cam

	classes = {0:'Meningioma', 1:'Glioma', 2:'Pituitary'}
	
	image_tensor, label, mask = data[0].type(torch.float), data[1], data[2]
	#convert image tensor to image
	image = image_tensor[0].clone().cpu().numpy()   # convert from GPU to CPU
	image = image.transpose((1,2,0))   # convert image back to height, weidth, channels
	imageio.imwrite(result_path+'/img%d.jpg' % (i + 1), image)

	mask = mask.squeeze().clone().cpu().numpy()
	mask = mask.transpose((1,2,0))
	imageio.imwrite(result_path+'/mask%d.jpg' % (i + 1), mask)
	
	logit = model(image_tensor)			
	h_x = F.softmax(logit, dim=1).data.squeeze()
	probs, idx = h_x.sort(0, True)
	print("True label : %d, Predicted label : %d, Probability : %.2f" % (label.item(), idx[0].item(), probs[0].item()))
	probs = probs.cpu().numpy()
	idx = idx.cpu().numpy()
		
	CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0].item()])

	# render the CAM and output
	print('The top1 prediction: %s' % classes[idx[0].item()])
	img = cv2.imread(result_path+'/img%d.jpg' % (i + 1))
	height, width, _ = img.shape

	heatmap = cv2.resize(CAMs[0],(width, height))

	plt.figure()
	plt.imshow(img)
	plt.imshow(heatmap, cmap='jet', alpha=0.5)
	plt.colorbar()
	plt.axis('off')
	plt.savefig(result_path+'/heatmap%d.jpg' % (i + 1))


	heatmap_mask = np.copy(heatmap)
	Q90 = np.quantile(heatmap_mask, .90)
	threshold = Q90
	heatmap_mask[heatmap_mask > threshold] = 255
	heatmap_mask[heatmap_mask <= threshold] = 0
	imageio.imwrite(result_path+'/heatmap_mask_Q90_%d.jpg' % (i + 1), heatmap_mask)

	heatmap_mask = np.copy(heatmap)
	Q95 = np.quantile(heatmap_mask, .95)
	threshold = Q95
	heatmap_mask[heatmap_mask > threshold] = 255
	heatmap_mask[heatmap_mask <= threshold] = 0
	imageio.imwrite(result_path+'/heatmap_mask_Q95_%d.jpg' % (i + 1), heatmap_mask)


	plt.figure()
	fig, [ax1, ax2, ax3, ax4] = plt.subplots(ncols=4)
	ax1.imshow(image)
	ax1.axis('off')
	ax1.set_title(classes[label.item()])
	ax2.imshow(img)
	ax2.imshow(heatmap, cmap='jet', alpha=0.5)
	ax2.set_title(classes[idx[0].item()]+'\n(Prob: %.2f)' % probs[0].item())
	ax2.axis('off')
	ax3.imshow(heatmap_mask, cmap='gray')
	ax3.axis('off')
	ax3.set_title('Mask Image')
	ax4.imshow(mask, cmap='gray')
	ax4.axis('off')
	ax4.set_title('Ground Truth')
	plt.savefig(result_path+'/CAM%d.jpg' % (i + 1))	


########## Visualization using Class Activation Mapping ##########

testset_file = 'dataset4.csv'
testset = Brain_data(testset_file, transform=None)
test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=1, shuffle=False)

#model = CNN()     # AlexNet
#model = vggNet16()
model = vggNet19()

if not os.path.exists('../BrainTumorClassification/final_result'):
	os.mkdir('../BrainTumorClassification/final_result')

result_path = '../BrainTumorClassification/final_result'

best_model = 'best_model_vgg19_4.save'
model.load_state_dict(torch.load(best_model))
#change the model to evaluation mode
model.eval()

for i, data in enumerate(test_loader):
	create_CAM(data, model, result_path, i)
