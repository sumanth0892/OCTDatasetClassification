import torch
import torchvision
import torch.nn as nn
import dataPrep as dP 
from getModel import CNN

#Import the data required 
trainLoader,testLoader = dP.getData()
neuralNet = CNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(neuralNet.parameters(),lr = 0.003,momentum = 0.9)

def trainModel(n_epochs,model,trainLoader,loss_fn,optimizer):
	for epoch in range(1,n_epochs+1):
		training_loss = 0.0
		for images,labels in trainLoader:
			optimizer.zero_grad()
			outputs = model(images)
			loss = loss_fn(outputs,labels)
			loss.backward()
			optimizer.step()
			training_loss += loss.item()
		print(epoch)
		print(training_loss/len(trainLoader))
		print("\n")
	
	for images,labels in testLoader:
		for i in range(len(labels)):
			img = images[i]
			c,w,h = img.shape
			img = img.view((1,c,w,h))
			true_label = labels.numpy()[i]
			with torch.no_grad():
				output = model(img)
			_,pred_label = torch.max(output.data,1)
			if true_label == pred_label:
				correctPredicted += 1
			totalPredicted += 1
	print("Testing accuracy:")
	print(correctPredicted*100/totalPredicted)
	



trainModel(10,neuralNet,trainLoader,loss_fn,optimizer)
			
