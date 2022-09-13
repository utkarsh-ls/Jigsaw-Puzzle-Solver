import torch
import torch.nn as nn
import torch.nn.functional as F
import types
from JigsawNetwork import Network

from dataset import ImageData


# Take input
cpu_cores_avail = 15#int(input("CPU cores: "))


# Get data
# base_path = 'data/mnist/'
base_path="data/ILSVRC2012_img_"
train_data = ImageData(base_path + 'train')
batch_size=64

train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=cpu_cores_avail)

val_data = ImageData(base_path + 'val')

val_loader = torch.utils.data.DataLoader(dataset=val_data,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=cpu_cores_avail)

test_data = ImageData(base_path + 'test')

test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=cpu_cores_avail)


# Get Device 
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device: ", device)

# Get Network
model = Network().to(device)
class Model(nn.Module):
    def __init__(self,model) -> None:
        super().__init__()
        self.features=model.features
        self.avgpool=model.avgpool

        for c in model.features.children():
            for params in  c.parameters():
                params.require_grad = False
        
        for c in model.avgpool.children():
            for params in  c.parameters():
                params.require_grad = False

        self.classifier=nn.Sequential(
        nn.Linear(82944,9* 1024),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(9*1024,4096),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 1000))
    
    def forward(self, x):
        B,T,C,H,W = x.size()
        x = x.transpose(0,1)

        x_list = []
        for i in range(9):
            z = self.features(x[i])
            z = self.avgpool(z)
            z = z.view([B,1,-1])
            x_list.append(z)

        x = torch.cat(x_list,1)
        # x = self.fc7(x.view(B,-1))
        x = self.classifier(x.view(B,-1))
        return x

# alex_model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
# model.classifier=nn.Sequential(
#         nn.Linear(82944,9* 1024),
#         nn.ReLU(inplace=True),
#         nn.Dropout(p=0.5),
#         nn.Linear(9*1024,4096),
#         nn.ReLU(inplace=True),
#         nn.Dropout(p=0.5),
#         nn.Linear(4096, 1000)
# )
# funcType = types.MethodType
# def forward_model(self, x):
#         B,T,C,H,W = x.size()
#         x = x.transpose(0,1)

#         x_list = []
#         for i in range(9):
#             z = self.features(x[i])
#             z = self.avgpool(z)
#             z = z.view([B,1,-1])
#             x_list.append(z)

#         x = torch.cat(x_list,1)
#         # x = self.fc7(x.view(B,-1))
#         x = self.classifier(x.view(B,-1))
        

#         return x
# type(model).forward = forward_model
# model = Model(alex_model).to(device)
# new_layers = nn.Sequential(

# )
# model.classifier = new_layers


# Hyperparameters
learning_rate = 0.001


# criterion
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# loop
epochs = 70
for epoch in range(epochs):
    for i, (batch_data, labels, original_data) in enumerate(train_loader):

        batch_data = batch_data.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(batch_data)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("Iteration: {}, Epoch: {}, Loss: {:.4f}".format(i+1, epoch + 1, loss.item()))

    
    print("Epoch [{}/{}], Loss: {:.4f}".format(epoch+1, epochs, loss.item()))
    
    # evaulate on validation set
    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        for batch_data, labels, original_data in val_loader:
            batch_data = batch_data.to(device)
            labels = labels.to(device)
            outputs = model(batch_data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the {} test images: {}\n'.format(total, 100 * correct / total))
        model.train()
