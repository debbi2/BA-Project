import os, code
import PIL
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image

class Brain_data(Dataset):
	def __init__(self, path, transform=None):
		lines = open(path).readlines()
		self.data = {line.strip().split(",")[0]:int(line.strip().split(",")[1]) for line in lines}     # create dictionary {data_name: label}
		self.list_of_data = list(self.data.keys())
		
		for d_name in self.list_of_data:
			label = self.data[d_name]
			image = Image.open('BrainData/image/{}.jpg'.format(d_name)).resize((224,224))    # image size = 224x224, image in PIL image
			mask = Image.open('BrainData/mask/{}.jpg'.format(d_name)).resize((224,224))    # mask size = 224x224, mask in PIL image
			self.data[d_name] = [image, label, mask]

		if transform is not None:
			self.transform = transforms.Compose([
				transforms.RandomRotation(degrees=(-90,90)),
				transforms.RandomHorizontalFlip(p=0.5),
				transforms.ToTensor()
				])
		else:
			self.transform = transforms.ToTensor()

	def __len__(self):
		return len(self.list_of_data)

	def __getitem__(self,idx):
		data = self.data[self.list_of_data[idx]]
		image, label, mask = data[0], data[1], data[2]
		
		image = self.transform(image)    # image.shape = [3, 224, 224]
		label = torch.tensor(label-1)    # original label was 1, 2, 3. Now set the label as 0, 1, 2
		mask = self.transform(mask)    # mask.shape = [3, 224, 224]
		
		return(image, label, mask)
