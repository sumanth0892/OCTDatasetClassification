import os
from torchvision import datasets,transforms 
from torch.utils.data import DataLoader

#Transforms for the image 
image_transforms = {
    # Train uses data augmentation
    'train':
    transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=200),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),

     # Test does not use augmentation
    'test':
    transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=200),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
traindir = '/home/bhargava/Documents/dataSets/OCTDataset/train/'
testdir = '/home/bhargava/Documents/dataSets/OCTDataset/test/'

def getData(traindir = traindir,testdir = testdir,batch_size = 128):
	#Replace the training and testing directories according to the new application.
	data = {
    'train':
    datasets.ImageFolder(root=traindir, transform=image_transforms['train']),
    'test':
    datasets.ImageFolder(root=testdir, transform=image_transforms['test'])
	}
	dataloaders = {
    'train': DataLoader(data['train'], batch_size=batch_size, shuffle=True),
    'test': DataLoader(data['test'], batch_size=batch_size, shuffle=True)
	}
	trainLoader = dataloaders['train']
	testLoader = dataloaders['test']
	return trainLoader,testLoader



