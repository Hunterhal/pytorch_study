import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim 

transform = transforms.Compose(          
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True,     
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)

classes = ('0', '1', '2', '3',                
           '4', '5', '6', '7', '8', '9')

from c_neural_network_file import*                                 #imports nn

criterion = nn.CrossEntropyLoss()                                  #loss function
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)    #adjusts parameters to minimize loss

for epoch in range(2):                                             # loop over the dataset 

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()                   #prevents gradients from adding up for every pass

        # forward + backward + optimize
        outputs = net(inputs)                   #nn inference
        loss = criterion(outputs, labels)       #puts outputs into loss function to calculate the loss between target and tested outputs 
        loss.backward()                         #computes gradients
        optimizer.step()                        #updates weights

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
PATH = 'mnist_net.pth'
torch.save(net.state_dict(), PATH)               #saves our model
