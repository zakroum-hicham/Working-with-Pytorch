import torch
from torch import nn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np

#################### Prepare the data
X,y = datasets.load_breast_cancer(return_X_y=True)

# split the data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=10)

# logitic reg needs the data to be Standardized
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# transform the data to tensor 
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))

y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

# 
y_train ,y_test = y_train.view(-1,1), y_test.view(-1,1)



class LogisticRegression(nn.Module):
    def __init__(self,input_dims) -> None:
        super(LogisticRegression,self).__init__()
        self.linear = nn.Linear(input_dims,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        return self.sigmoid(self.linear(x))


model = LogisticRegression(X.shape[1])
lossfn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)


# loop

epoches = 200

for i in range(epoches):
    y_predicted = model(X_train)
    loss = lossfn(y_predicted,y_train)

    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    print(f"epoche = {i}, loss={loss.item():.4f}")



with torch.no_grad():

    prediction = model(X_test)

    print(f"acc = {accuracy_score(y_test,prediction.round()):.4f}")