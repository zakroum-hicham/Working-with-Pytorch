import torch
from torch import nn
from sklearn import datasets
import numpy as np

# 0) prepare Data

X,Y  = datasets.make_regression(100,1,noise=10,random_state=10)
X,Y = torch.from_numpy(X.astype(np.float32)),torch.from_numpy(Y.astype(np.float32))
Y = Y.view(-1,1)

# 1) design the model
class LinearRegression(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(LinearRegression,self).__init__()
        self.linear = nn.Linear(input_dim,output_dim)

    def forward(self,x):
        return self.linear(x)
    


model = LinearRegression(1,1)

# 2) loss and the optimizer
lossFn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)


# 3) training loop 

epchos = 200
for i in range(epchos):
    # forward 
    y_predicted = model(X)
    # loss
    loss = lossFn(y_predicted,Y)
    # backward
    loss.backward()
    # update
    optimizer.step()
    optimizer.zero_grad()

    # log data

    if i%10 == 0:
        print(f"epoche = {i}, loss = {loss.item():.4f}, ")

# plot
import matplotlib.pyplot as plt

prediction  = model(X).detach().numpy()

plt.plot(X.numpy(),Y.numpy(),"ro")
plt.plot(X.numpy(),prediction,"b")
plt.show()
