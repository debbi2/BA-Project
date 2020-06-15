import os, glob
import PIL
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
from PIL import Image
from torch import nn
from torch.nn import CrossEntropyLoss, Softmax
from torch.optim import SGD
import time, code
from models_vgg import vggNet16, vggNet19
from data_process import Brain_data



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data_choice = 4

trainset_file_list = glob.glob('dataset*.csv')
testset_file = 'dataset{}.csv'.format(data_choice)
trainset_file_list.remove(testset_file)

with open('trainset.csv', 'w') as outfile:
	for file_name in trainset_file_list:
		with open(file_name) as infile:
			outfile.write(infile.read())

trainset_file = 'trainset.csv'

trainset = Brain_data(trainset_file, transform=True)
testset = Brain_data(testset_file, transform=None)

train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=1, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=1, shuffle=False)


########## Model Training ##########
class_names = ['Meningioma', 'Glioma', 'Pitutary']
model = vggNet16()
#model = vggNet19()

# freeze layers
for param in model.features.parameters():
	param.require_grad = False


def train(model, train_loader, criterion, optimizer, epoch, total_epoch, log_step):
	model.train()

	running_loss = 0
	total_size = 0
	correct = 0
	print('TRAINING:')
	for i,(image, label, mask) in enumerate(train_loader):
		image, label = image.to(device,dtype=torch.float), label.to(device)
		optimizer.zero_grad()
		output = model(image)
		loss = criterion(output, label)
		running_loss += loss.item()
		total_size += image.size(0)

		predicted = torch.argmax(output, dim=1)
		correct += predicted.eq(label.view_as(predicted)).long().cpu().sum().item()

		loss.backward()
		optimizer.step()

		if (i + 1) % log_step == 0:
			print('Epoch [%d/%d], Iter [%d/%d], Average Loss: %.4f, Acc: %.4f (%d/%d)'
				% (epoch, total_epoch,
					i + 1, len(train_loader),
					running_loss / total_size,
					correct / total_size, correct, total_size))

	running_loss /= len(train_loader)

	return running_loss, correct/len(train_loader)



def evaluate(model, val_loader, criterion, epoch, total_epoch, log_step, min_loss):
	model.eval()

	running_loss =0
	total_size = 0
	correct = 0
	print('VALIDATION:')
	for i,(image, label, mask) in enumerate(test_loader):
		image, label = image.to(device,dtype=torch.float), label.to(device)
		output = model(image)
		loss = criterion(output, label)
		running_loss += loss.item()
		total_size += image.size(0)

		predicted = torch.argmax(output, dim=1)
		correct += predicted.eq(label.view_as(predicted)).long().cpu().sum().item()

		if (i + 1) % log_step == 0:
			print('Epoch [%d/%d], Iter [%d/%d], Average Loss: %.4f, Acc: %.4f (%d/%d)'
				% (epoch, total_epoch,
					i + 1, len(test_loader),
					running_loss / total_size,
					correct / total_size, correct, total_size))

	running_loss /= len(test_loader)
	if running_loss < min_loss:
		torch.save(model.state_dict(),"best_model_vgg16_{}.save".format(data_choice))
		print('Best model with loss of {:.6f} is saved.'.format(running_loss))

	return running_loss, correct/len(test_loader)


# set model to run on GPU
model = model.to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=0.0003)
epochs = 50
log_step = 100
min_loss = 999

total_train_loss, total_train_acc = [], []
total_val_loss, total_val_acc = [], []

start_training = time.time()
for epoch in range(epochs):
	start_time = time.time()

	train_loss, train_acc = train(model, train_loader, criterion, optimizer, epoch+1, epochs, log_step)
	total_train_loss.append(train_loss)
	total_train_acc.append(train_acc)

	val_loss, val_acc = evaluate(model, test_loader, criterion, epoch+1, epochs, log_step, min_loss)
	if val_loss < min_loss:
		min_loss = val_loss
	total_val_loss.append(val_loss)
	total_val_acc.append(val_acc)

	time_elapsed = time.time() - start_time
	print('Train Loss: {:.4f} | Train Accuracy: {:.4f} | Validation Loss: {:.4f} | Validation Accuracy: {:.4f}'.format(
		train_loss, train_acc, val_loss, val_acc))
	print('Elapsed time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

training_time = time.time() - start_training
print('Training took {:.0f}m {:.0f}s'.format(training_time // 60, training_time % 60))
print('Training Loss: ', total_train_loss)
print('Validation Loss: ', total_val_loss)
print('Training Accuracy: ', total_train_acc)
print('Validation Accuracy: ', total_val_acc)


########## Plot Loss and Accuracy Graphs ##########
plt.figure()
plt.plot(total_train_loss, label='Training loss')
plt.plot(total_val_loss, label='Validation loss')
plt.title('Loss Metrics')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()
plt.savefig('Loss_Metrics_vgg16_{}.jpg'.format(data_choice))

plt.figure()
plt.plot(total_train_acc, label='Training Accuracy')
plt.plot(total_val_acc, label='Validation Accuracy')
plt.title('Accuracy Metrics')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.savefig('Accuracy_Metrics_vgg16_{}.jpg'.format(data_choice))
##################################################
