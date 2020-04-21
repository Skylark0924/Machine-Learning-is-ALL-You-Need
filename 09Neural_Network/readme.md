# Neural Network
> Classification & Regression

## Principle
![](https://gblobscdn.gitbook.com/assets%2F-LYtTdBhaX9EsGjMVkHV%2F-LfB3qVJsT2tC62oMk6R%2F-LfB3w0x_CIhlmmHWcUb%2Fimage.png?alt=media&token=fbefd92a-e896-45bc-b422-601566696db4)

神经网络的原理就不过多介绍了，看上图就好。

## PyTorch五种网络搭建方式
### From Scratch
用torch构建weight, bias
```
import torch
import torch.nn.functional as F

# generating some random features
features = torch.randn(1, 16) 

# define the weights
W1 = torch.randn((16, 12), requires_grad=True)
W2 = torch.randn((12, 10), requires_grad=True)
W3 = torch.randn((10, 1), requires_grad=True)

# define the bias terms
B1 = torch.randn((12), requires_grad=True)
B2 = torch.randn((10), requires_grad=True)
B3 = torch.randn((1), requires_grad=True)

# calculate hidden and output layers
h1 = F.relu((features @ W1) + B1)
h2 = F.relu((h1 @ W2) + B2)
output = torch.sigmoid((h2 @ W3) + B3)
```

### 继承 nn.Module class
__init__ + forward 结构

```
import torch
import torch.nn.functional as F
from torch import nn

# define the network class
class MyNetwork(nn.Module):
    def __init__(self):
        # call constructor from superclass
        super().__init__()
        
        # define network layers
        self.fc1 = nn.Linear(16, 12)
        self.fc2 = nn.Linear(12, 10)
        self.fc3 = nn.Linear(10, 1)
        
    def forward(self, x):
        # define forward pass
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# instantiate the model
model = MyNetwork()

# print model architecture
print(model)
```

### Using torch.nn.Sequential
```
from torch import nn

# define model architecture
model = nn.Sequential(
    nn.Linear(16, 12),
    nn.ReLU(),
    nn.Linear(12, 10),
    nn.ReLU(),
    nn.Linear(10, 1),
    nn.Sigmoid()
)

# print model architecture
print(model)
```

### Mixed Approach
```
import torch
from torch import nn

class MyNetwork2(nn.Module):
    def __init__(self):
        super().__init__()
        
        # define the layers
        self.layers = nn.Sequential(
            nn.Linear(16, 12),
            nn.ReLU(),
            nn.Linear(12, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
    
    def forward(self, x):
        # forward pass
        x = torch.sigmoid(self.layers(x))
        return x

# instantiate the model
model = MyNetwork2()

# print model architecture
print(model)
```

### 内部实例化型
这也是**本文采用的方式**，以便于匹配sklearn和keras的fit, predict流程。取消了外部的实例化，将训练和预测过程包含在类中。
```
class Torch_NN(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super().__init__()
        # define model architecture
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('gpu')
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], num_classes),
            nn.Sigmoid()
        ).to(self.device)
        
        print('Model:\n{}\nDevice: {}'.format(self.model, self.device))
    
    def fit(self, X_train, Y_train, epochs, batch_size, learning_rate):
        dtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
        tensor_x = torch.Tensor(X_train) # transform from array to torch tensor
        tensor_y = torch.Tensor(Y_train)
        MyDataset = data.TensorDataset(tensor_x, tensor_y) # making the dataset
        # 数据加载器 DataLoader
        # 训练数据加载器
        train_loader = data.DataLoader(
            dataset=MyDataset, 
            batch_size=batch_size, shuffle=True)
        total_step = len(train_loader)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        for epoch in range(epochs):
            for i, (x, labels) in enumerate(train_loader):
                x = x.to(self.device, dtype= torch.float)
                labels = labels.to(self.device, dtype= torch.long)

                # 前向传播
                outputs = self.model(x)
                loss = criterion(outputs, labels)

                # 反向传播并优化
                optimizer.zero_grad()  # 注意每步迭代都需要清空梯度缓存
                loss.backward()
                optimizer.step()

                if (i+1) % 30 == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                            .format(epoch+1, epochs, i+1, total_step, loss.item()))

    def predict(self, X_test):
        X_test = torch.Tensor(X_test).to(self.device)
        Y_pred = self.model(X_test)
        _, Y_pred = torch.max(Y_pred.data, 1)
        return Y_pred.cpu().detach().numpy()
```

## Self-implement 
### Bulid Network
按照 $y=wx+b$ 的形式构建，当然你也可以按 $y=x^Tw+b$，注意维度就可以了。
```
    def feedforward(self):
        self.layer1 = relu(np.dot(self.w1, self.X))
        self.layer2 = relu(np.dot(self.w2, self.layer1))
        self.layer3 = sigmoid(np.dot(self.w3, self.layer2))
        return self.layer3
```

### Backpropgation
**链式法则**

1. 先计算最后一层权重$w_3$的更新方向：
   $$ y_{out} = sigmoid(w_3h_2+b3) $$
   $$Loss(\hat{y}_{out}, y_{label}) = \sum^n_{i=1}(\hat{y}_-y)^2$$
   $$ \frac{\partial Loss(y, \hat{y})}{\partial w_3}=\frac{\partial {Loss}(y, \hat{y})}{\partial \hat{y}} * \frac{\partial \hat{y}}{\partial z_3} * \frac{\partial z_3}{\partial w_3} \quad \text { where } z_3=w_3 h_2+b_3 $$
2. 权重$w_2$的更新方向:
   $$h_2=relu(w_2h_1+b_2)=relu(z_2)$$
   $$\frac{\partial Loss(y, \hat{y})}{\partial w_2}=\frac{\partial {Loss}(y, \hat{y})}{\partial \hat{y}} * \frac{\partial \hat{y}}{\partial z_3} * \frac{\partial z_3}{\partial h_2} * \frac{\partial h_2}{\partial z_2} * \frac{\partial z_2}{\partial w_2} \\ \text { where } z_2=w_2 h_1+b_2$$
3. 权重$w_1$的更新方向:
   $$h_1=relu(w_1x+b_1)=relu(z_1)$$
   $$\frac{\partial Loss(y, \hat{y})}{\partial w_1}=\frac{\partial {Loss}(y, \hat{y})}{\partial \hat{y}} * \frac{\partial \hat{y}}{\partial z_3} * \frac{\partial z_3}{\partial h_2} * \frac{\partial h_2}{\partial z_2} * \frac{\partial z_2}{\partial h_1}* \frac{\partial h_1}{\partial z_1} * \frac{\partial z_1}{\partial w_1}\\ \text { where } z_1=w_1 x+b_1$$
```
    def backprop(self):
        input_data = self.X
        temp3 = 2*(self.Y - self.output) * sigmoid_derivative(self.output)
        d_w3 = np.dot(temp3, self.layer2.T)
        temp2 = np.dot(self.w3.T, temp3) * relu_derivative(self.layer2)
        d_w2 = np.dot(temp2, self.layer1.T)
        temp1 = np.dot(self.w2.T, temp2) * relu_derivative(self.layer1)
        d_w1 = np.dot(temp1, input_data.T)

        # Update parameters
        self.w1 += self.learning_rate * d_w1
        self.w2 += self.learning_rate * d_w2   
        self.w3 += self.learning_rate * d_w3 
```

## Reference
1. [Three Ways to Build a Neural Network in PyTorch](https://towardsdatascience.com/three-ways-to-build-a-neural-network-in-pytorch-8cea49f9a61a)