## MINIST手写数字数据集--神经网络（mini-batch）
### 1.1 数据集介绍
   MNIST 数据集主要由一些手写数字的图片和相应的标签组成，图片一共有10 类，分别对应从0～9 ，共10 个阿拉伯数字。
### 1.2 思路介绍
- 导入数据集
- 对导入数据进行处理
- 设计神经网路
- 训练的同时调整超参数得到正确率较高的模型
- 保存模型
- 对测试集中数据进行预测并将结果保存到新文件
### 1.3 分析数据集并读入数据处理
  数据以.CSV形式存储在MNIST_dataset文件夹下，训练数据集共有42000个数据，测试集有28000个数据。  

  **1. 首先定义文件读取函数**
```python
def read_csv(fileRoad):
    return np.loadtxt(fileRoad, dtype=str, skiprows=1)
```
由于读取到的数据为字符串数组，一行即为一个图片1 * 784，784源于图像大小 28 * 28=74采用以下代码将字符串变为浮点列表（便于进行转换为ndarray或者tensor进行后续处理）
```python
data = [float(i) for i in datas[index].split(',')]
```
 **2. 定义图片索引查看和拼接全图查看函数**
```python
# 按索引查看图片
def img_show(fileRoad,index):
    datas = read_csv(fileRoad)
    data = [float(i) for i in datas[index].split(',')]
    if len(data) == 785:
        content = data[1:]
    elif len(data) == 784:
        content = data
    img = np.array(content).reshape(28, 28)
    pilImg = Image.fromarray(np.uint8(img))
    pilImg.show()

# 对图片进行拼接，全图显示
def img_show(fileRoad):
    datas = read_csv(fileRoad)
    dataAllNum = len(datas)
    columNum = int(np.sqrt(dataAllNum))
    rowNum = int(dataAllNum/columNum) + 1
    count = 0
    img = []
    for k in range(rowNum):
        row = []
        for n in range(columNum):
            if count<dataAllNum:
                data = [float(i) for i in datas[count].split(',')]
                if len(data) == 785:
                    content = data[1:]
                elif len(data) == 784:
                    content = data
            else:
                content = np.ones([1,784],dtype=np.float)
            imgSingle = np.array(content).reshape(28, 28)
            if len(row)==0:
                row.append(imgSingle)
            else:
                row[0] = np.hstack((row[0],imgSingle))
            count = count + 1
        if len(img)==0:
            img.append(row[0])
        else:
            img[0] = np.vstack((img[0],row[0]))
    pilImg = Image.fromarray(np.uint8(img[0]))
    pilImg.show()
```

 <div  align="center">    
<img src="https://github.com/toygerse/MINIST_CLARIFY/blob/master/trainImg.png" alt="训练集" width="500" height="313" align="middle" />

  <div align="center"><I>训练集全图片显示</I></div>
</div>
  
  
 <div  align="center">    
<img src="https://github.com/toygerse/MINIST_CLARIFY/blob/master/testImg.png" alt="" width="500" height="313" align="middle" />

  <div  align="center"><I>测试集全图片显示</I></div>
</div>  

  **3. 定义dataset和dataloader**  
        
```python
class Mydata(Dataset):
    def __init__(self, dir):
        self.datas = read_csv(dir)
        self.rootDir = dir

    def __getitem__(self, index):
        data = [float(i) for i in self.datas[index].split(',')]
        if len(data)==785:
            label = int(data[0])
            sample = torch.tensor(data[1:])
            return sample,label
        elif len(data)==784:
            sample = torch.tensor(data)
            return sample

    def __len__(self):
        return len(self.datas)

```

```python
trainLoader = DataLoader(trainData,batch_size=150,shuffle=True)
```
其中batch_size为超参数，在测试后选取batch为150
### 1.4 设计神经网络  

```python
class Net(nn.Module):
    def __init__(self,in_num,out_num):
        super(Net,self).__init__()
        self.network = nn.Sequential(
            nn.Linear(in_num, 1000),
            nn.ReLU(),
            nn.Linear(1000, 800),
            nn.Sigmoid(),
            nn.Linear(800, 10),
            nn.Softmax(dim=1)
        )
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.08)
        self.loss_func = torch.nn.CrossEntropyLoss()

    def forward(self,x):
        return self.network(x)
```
### 1.5 训练神经网络
设置10个epoch
```python
trainData = Mydata(trainRoad)
trainLoader = DataLoader(trainData,batch_size=150,shuffle=True)
myNet = Net(784,10)
epochList = []
lossList = []
count = 0
print("can't find MINIST_model, start training...")
for epoch in range(1,11):
  correctNum = 0
  dataNum = 0
  lossNum = 0
  for inputs,label in trainLoader:
    predY = myNet.forward(inputs)
    loss = myNet.loss_func(predY,label)
    count_accuracy(torch.max(predY, 1)[1].numpy(),label.numpy())
    lossNum = lossNum + loss.item()
    myNet.optimizer.zero_grad()
    loss.backward()
    myNet.optimizer.step()
  lossList.append(float(lossNum) / dataNum)
  epochList.append(epoch-1)
  accuracy = float(correctNum*100)/dataNum
  print('epoch:%d | accuracy = %.2f%%' % (epoch, accuracy))

print("training complete")
```
训练结束，对模型进行保存（准确率高于95%）
```python
if accuracy>=95:
  torch.save(myNet, 'MINIST_model.pkl')
  print("model saved")
else:
  print("accuracy is too low,didn't saved this model")
```
用 Matplotlib对10次epoch的loss绘制图像
```python
plt.plot(epochList,lossList)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()
```
 <div  align="center">    
<img src="https://github.com/toygerse/MINIST_CLARIFY/blob/master/MINIST_EPOCH10.png" alt="训练集" width="500" height="313" align="middle" />

  <div align="center"><I>epoch-loss函数</I></div>
</div>  

发现随着epoch增加loss下降越来越不明显，增加epoch对改进模型效果不大  

### 1.5 完整代码
  
```python
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

dataNum = 0
correctNum = 0

testRoad = "./MNIST_dataset/test.csv"
trainRoad = "./MNIST_dataset/train.csv"
sampleRoad = "./MNIST_dataset/sample_submission.csv"
predictRoad = "./MNIST_dataset/sample_submission_predict.csv"

def read_csv(fileRoad):
    return np.loadtxt(fileRoad, dtype=str, skiprows=1)

def img_show(fileRoad,index):
    datas = read_csv(fileRoad)
    data = [float(i) for i in datas[index].split(',')]
    if len(data) == 785:
        content = data[1:]
    elif len(data) == 784:
        content = data
    img = np.array(content).reshape(28, 28)
    pilImg = Image.fromarray(np.uint8(img))
    pilImg.show()

def img_show(fileRoad):
    datas = read_csv(fileRoad)
    dataAllNum = len(datas)
    columNum = int(np.sqrt(dataAllNum))
    rowNum = int(dataAllNum/columNum) + 1
    count = 0
    img = []
    for k in range(rowNum):
        row = []
        for n in range(columNum):
            if count<dataAllNum:
                data = [float(i) for i in datas[count].split(',')]
                if len(data) == 785:
                    content = data[1:]
                elif len(data) == 784:
                    content = data
            else:
                content = np.ones([1,784],dtype=np.float)
            imgSingle = np.array(content).reshape(28, 28)
            if len(row)==0:
                row.append(imgSingle)
            else:
                row[0] = np.hstack((row[0],imgSingle))
            count = count + 1
        if len(img)==0:
            img.append(row[0])
        else:
            img[0] = np.vstack((img[0],row[0]))
    pilImg = Image.fromarray(np.uint8(img[0]))
    pilImg.show()

def count_accuracy(pred_y, label):
    global dataNum, correctNum
    for i in range(len(label)):
        dataNum =dataNum +1
        if pred_y[i] == label[i]:
            correctNum = correctNum +1

def test_existModle(myNet):
    trainData = Mydata(trainRoad)
    trainLoader = DataLoader(trainData, batch_size=150, shuffle=True)
    for input,label in trainLoader:
        output = myNet(input)
        count_accuracy(torch.max(output, 1)[1].numpy(), label.numpy())
    print('accuracy = %.2f%%' % (float(correctNum * 100) / dataNum))

def predict(myNet):
    try:
        pd.read_csv(predictRoad, encoding='utf-8')
        print("file sample_submission_predict.csv exist")
    except:
        sampleForm = pd.read_csv(sampleRoad, encoding='utf-8')
        dataset =Mydata(testRoad)
        testdata = DataLoader(dataset,batch_size=1,shuffle=False)
        print("start to predict sample_submission.csv and save...")
        for i,data in enumerate(testdata):
            output = myNet(data)
            result = torch.max(output, 1)[1].item()
            sampleForm['Label'].loc[i] = result
        sampleForm.to_csv(predictRoad, encoding='utf-8', index=False)
        print("complete save sample_submission_predict.csv")


class Mydata(Dataset):
    def __init__(self, dir):
        self.datas = read_csv(dir)
        self.rootDir = dir

    def __getitem__(self, index):
        data = [float(i) for i in self.datas[index].split(',')]
        if len(data)==785:
            label = int(data[0])
            sample = torch.tensor(data[1:])
            return sample,label
        elif len(data)==784:
            sample = torch.tensor(data)
            return sample

    def __len__(self):
        return len(self.datas)

class Net(nn.Module):
    def __init__(self,in_num,out_num):
        super(Net,self).__init__()
        self.network = nn.Sequential(
            nn.Linear(in_num, 1000),
            nn.ReLU(),
            nn.Linear(1000, 800),
            nn.Sigmoid(),
            nn.Linear(800, 10),
            nn.Softmax(dim=1)
        )
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.08)
        self.loss_func = torch.nn.CrossEntropyLoss()

    def forward(self,x):
        return self.network(x)

if __name__ == '__main__':
    try:
        myNet = torch.load('MINIST_model.pkl')
        print("MINIST_model exist and have been loaded")
        print("testing accuracy...")
        test_existModle(myNet)
        predict(myNet)
    except:
        trainData = Mydata(trainRoad)
        trainLoader = DataLoader(trainData,batch_size=150,shuffle=True)
        myNet = Net(784,10)
        epochList = []
        lossList = []
        count = 0
        print("can't find MINIST_model, start training...")
        for epoch in range(1,11):
            correctNum = 0
            dataNum = 0
            lossNum = 0
            for inputs,label in trainLoader:
                predY = myNet.forward(inputs)
                loss = myNet.loss_func(predY,label)
                count_accuracy(torch.max(predY, 1)[1].numpy(),label.numpy())
                lossNum = lossNum + loss.item()
                myNet.optimizer.zero_grad()
                loss.backward()
                myNet.optimizer.step()
            lossList.append(float(lossNum) / dataNum)
            epochList.append(epoch-1)
            accuracy = float(correctNum*100)/dataNum
            print('epoch:%d | accuracy = %.2f%%' % (epoch, accuracy))

        print("training complete")
        if accuracy>=95:
            torch.save(myNet, 'MINIST_model.pkl')
            print("model saved")
        else:
            print("accuracy is too low,didn't saved this model")
        plt.plot(epochList,lossList)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.show()


```
