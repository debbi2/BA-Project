import os
import PIL
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from skimage import io, transform
from torch import nn
from torch.nn import Softmax
import time, code
from models_vgg import vggNet16, vggNet19
from models_alexnet import CNN
from data_process import Brain_data
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix


def performance_report(model_name, class_names):
	if model_name == 'alexnet':
		model = CNN()
	elif model_name == 'vgg16':
		model = vggNet16()
	elif model_name == 'vgg19':
		model = vggNet19()

	file_out = open('{}_performance.csv'.format(model_name),'w')

	for data_num in range(5):
		testset_file = 'dataset{}.csv'.format(data_num)
		testset = Brain_data(testset_file, transform=None)
		test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=1, shuffle=False)

		best_model = 'best_model_{}_{}.save'.format(model_name, data_num)
		model.load_state_dict(torch.load(best_model))
		#change the model to evaluation mode
		model.eval()

		true_label = []
		prediction = []
		for i,(image, label, mask) in enumerate(test_loader):
			output = model(image)
			probs = nn.Softmax(dim=1)(output)
			value, pred = torch.max(probs, dim=1)
			prediction += list(pred.numpy())
			true_label += list(label.numpy())

		report = classification_report(true_label, prediction, target_names=class_names)
		file_out.write('Iteration {}\n'.format(data_num+1) + str(report) + '\n')



def plot_confusion_matrix(cm, class_names, title='Confusion Matrix', cmap=plt.cm.Blues):
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(class_names))
	plt.xticks(tick_marks, class_names, rotation=45)
	plt.yticks(tick_marks, class_names)
	plt.tight_layout()
	plt.xlabel('Predicted label')
	plt.ylabel('True label')
	plt.show()



def confusion_metrix(model_name, data_num, class_names):
	if model_name == 'alexnet':
		model = CNN()
	elif model_name == 'vgg16':
		model = vggNet16()
	elif model_name == 'vgg19':
		model = vggNet19()

	testset_file = 'dataset{}.csv'.format(data_num)
	testset = Brain_data(testset_file, transform=None)
	test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=1, shuffle=False)

	best_model = 'best_model_{}_{}.save'.format(model_name, data_num)
	model.load_state_dict(torch.load(best_model))
	#change the model to evaluation mode
	model.eval()

	true_label = []
	prediction = []
	for i,(image, label, mask) in enumerate(test_loader):
		output = model(image)
		probs = nn.Softmax(dim=1)(output)
		value, pred = torch.max(probs, dim=1)
		prediction += list(pred.numpy())
		true_label += list(label.numpy())

	cm = confusion_matrix(true_label, prediction)
	np.set_printoptions(precision=2)
	print(cm)
	plt.figure()
	plot_confusion_matrix(cm, class_names)
	plt.savefig('confusion_matrix.jpg')

	cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	print(cm_normalized)
	plt.figure()
	plot_confusion_matrix(cm_normalized, class_names)
	plt.savefig('normalized_confusion_matrix.jpg')

	'''
	cm = [[139   2   0]
		 [  0 285   0]
		 [  5   3 178]]
	cm_normalized = [[0.99 0.01 0.  ]
					[0.   1.   0.  ]
					[0.03 0.02 0.96]]
	'''






class_names = ['Meningioma', 'Glioma', 'Pitutary']
models = ['alexnet', 'vgg16', 'vgg19']

for model in models:
	performance_report(model, class_names)


confusion_metrix('vgg19', 4, class_names)