import torch
import torch.optim as optim
from torch import nn
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


# device config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# parameters 
batch_size = 64
learning_rate = 0.01
epochs = 4

# Prepare the Data 

transform = transforms.Compose(
    [
    transforms.ToTensor(),
    transforms.Normalize(0,1)
    ]
)

train_data = datasets.FashionMNIST("data",train=True,transform=transform,download=True)
test_data = datasets.FashionMNIST("data",train=False,transform=transform,download=True)

out_dims = len(train_data.classes)


train_data_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
test_data_loader = DataLoader(test_data,batch_size=batch_size,shuffle=False)


# build the model


class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1,16,(3,3),padding=1)
        self.pool = nn.MaxPool2d((2,2))
        self.conv2 = nn.Conv2d(16,32,(3,3),padding=1)
        self.conv3 = nn.Conv2d(32,64,(3,3),padding=1)
        self.linear = nn.Linear(576,10) #576 = 64*3*3
    
    def forward(self,x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        x = torch.flatten(x,1)
        x = self.linear(x)
        return x



model = CNN()


# loss , optimizer

lossFn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate)


# traing loop

for epoch in range(epochs):
    correct,lossT = 0,0
    accuracy = 0
    batch_count = len(train_data_loader)

    n = 0

    # Initialize tqdm
    pbar = tqdm(enumerate(train_data_loader), total=batch_count, 
                desc=f"Epoch {epoch + 1}/{epochs} - accuracy: {accuracy:.2f}%, loss: {lossT:.4f}")

    for i,(images,labels) in pbar:
        logits = model(images)

        loss = lossFn(logits,labels)
        lossT+=loss
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        # Calculate the accuracy
        prediction = torch.argmax(logits, dim=1)
        n += labels.size(0)
        correct += (prediction == labels).sum()
        accuracy = (100.0 * correct) / n

        
        # Update tqdm description
        if (i + 1) % 10 == 0:
            pbar.set_description(
                f"Epoch {epoch + 1}/{epochs} - accuracy: {accuracy:.2f}%, loss: {lossT/(i+1):.4f}"
            )



# test
print("testing with Test data ....")
with torch.no_grad():
    correct = 0
    n = 0
    for images,labels in test_data_loader:

        logits = model(images)

        prediction = torch.argmax(logits,1)
        n+= labels.size(0)
        correct += (prediction == labels).sum()

    accuracy = (100.0 * correct) / n

    print("Test data : ")
    print(f"Accuracy = {accuracy:.4f}%")


# plot same samples
import matplotlib.pyplot as plt


print("ploting same simples ...")
with torch.no_grad():
        for i,(images,labels) in enumerate(test_data_loader):
            images = images[:8]
            labels = labels[:8]

            # 
            logits = model(images)

            prediction = torch.argmax(logits,1)

            plt.figure(figsize=(10, 8))
            for ii in range(8):
                ax = plt.subplot(4,2,ii+1)
                ax.set_title(f"predicted as {train_data.classes[prediction[ii]]}, correct {train_data.classes[labels[ii]]}")
                ax.imshow(images[ii].reshape(28,28).cpu().numpy(),cmap="gray")
                ax.axis("off")

            plt.tight_layout()
            plt.show()
            break