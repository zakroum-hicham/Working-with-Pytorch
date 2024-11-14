import torch
from torch import nn
import torch.utils
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from tqdm import tqdm

# device configuration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# params

batches = 32
input_dim = 28*28
output_dim = 10
hidden_layers = [32,16,32]
learning_rate = 0.01
epochs = 2

# prepare the data

composer  = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(0,1)
    ]
)

train_data = datasets.MNIST("./data",transform=composer,train=True,download=True)
test_data = datasets.MNIST("./data",transform=composer,train=False)

train_data = DataLoader(train_data,batch_size=batches,shuffle=True)
test_data = DataLoader(test_data,batch_size=batches,shuffle=False)


# train_data_iter = iter(train_data)
# print(next(train_data_iter)[0][0])


# model class

class MLP(nn.Module):
    def __init__(self,input_dim:int,hidden_layers:list,output_dim:int)-> None:
        super().__init__()
        if len(hidden_layers) != 3:
            raise Exception("The hidden_layers List should be of len 3")
        self.linear1 = nn.Linear(input_dim,hidden_layers[0])
        self.linear2 = nn.Linear(hidden_layers[0],hidden_layers[1])
        self.linear3 = nn.Linear(hidden_layers[1],hidden_layers[2])
        self.linear4 = nn.Linear(hidden_layers[2],output_dim)

    def forward(self,x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        x = self.linear4(x)
        return x

model = MLP(input_dim,hidden_layers,output_dim)

# loss and optimzer
lossFn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)



# training loop

for epoch in range(epochs):
    correct = 0
    n = 0
    accuracy = 0
    total_loss = 0 
    batch_count = len(train_data)

    # Initialize tqdm
    pbar = tqdm(enumerate(train_data), total=batch_count, 
                desc=f"Epoch {epoch + 1}/{epochs} - accuracy: {accuracy:.2f}%, loss: {total_loss:.4f}")

    for i, (images, labels) in pbar:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        pred = model(images)
        loss = lossFn(pred, labels)
        loss.backward()
        optimizer.step()

        _, prediction = torch.max(pred, dim=1)
        n += labels.size(0)
        correct += (prediction == labels).sum().item()
        accuracy = (100.0 * correct) / n

        # Accumulate the loss
        total_loss += loss.item()

        # Update tqdm description
        if (i + 1) % 100 == 0:
            pbar.set_description(
                f"Epoch {epoch + 1}/{epochs} - accuracy: {accuracy:.2f}%, loss: {total_loss / (i + 1):.4f}"
            )


# test

with torch.no_grad():
    correct = 0
    n = 0
    for images,labels in test_data:
        images = images.reshape(-1,28*28).to(device)
        labels.to(device)

        # 
        pred = model(images)

        _,prediction = torch.max(pred,1)
        n+= labels.size(0)
        correct += (prediction == labels).sum().item()

    accuracy = (100.0 * correct) / n

    print("Test data : ")
    print(f"Accuracy = {accuracy:.4f}%")


# plot same samples
import matplotlib.pyplot as plt


print("ploting same simples ...")
with torch.no_grad():
        for i,(images,labels) in enumerate(test_data):
            images = images[:8]
            labels = labels[:8]
            images = images.reshape(-1,28*28).to(device)
            labels.to(device)

            # 
            pred = model(images)

            _,prediction = torch.max(pred,1)

            plt.figure(figsize=(10, 8))
            for ii in range(8):
                ax = plt.subplot(4,2,ii+1)
                ax.set_title(f"predicted as {labels[ii]}")
                ax.imshow(images[ii].reshape(28,28).cpu().numpy(),cmap="gray")
                ax.axis("off")

            plt.tight_layout()
            plt.show()
            break