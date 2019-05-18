#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
from collections import OrderedDict
import copy


# In[4]:


mnist = np.loadtxt('mnist.csv', delimiter=',')


# In[ ]:


def train_test_split(csv_dataset):
    '''
    csv_dataset의 shape는 (10000, 785)이다.
    총 만 개의 데이터가 있고, 각 데이터는 레이블값(1개), 픽셀값(784개)들로 이루어져 있다.
    train_set 개수 : test_set 개수 = 80 : 20 의 비율로 데이터를 분할하고 레이블값과 픽셀값으로 한 번 더 분할하면,
    train_X.shape = (8000, 784)
    train_T.shape = (8000, 1)
    test_X.shape = (2000, 784)
    test_T.shape = (2000, 1) 이다.
    '''
    train_X = csv_dataset[:8000, 1:]
    train_X /= 256
    train_T = csv_dataset[:8000, 0]
    test_X = csv_dataset[8000:, 1:]
    test_X /= 256
    test_T = csv_dataset[8000:, 0]  
    
    return train_X, train_T, test_X, test_T


# In[15]:


def one_hot_encoding(T): # T is data의 label
    one_hot_label = np.zeros([T.shape[0],10])
    T = T.astype(np.uint8)
    one_hot_label[np.arange(T.shape[0]), T] = 1
    
    '''
    먼저 [데이터개수, 10] 크기의 배열을 만든다
    각 데이터마다의 레이블값과 같은 인덱스열에 1의 값을 넣어주어야 하는데,
    T에 들어있는 값은 float형이기 때문에, 형변환을 하지 않고 3번째 줄을 실행하면
    IndexError: arrays used as indices must be of integer (or boolean) type 와 같은 에러가 발생한다.
    그래서 np.astype을 이용해 int형으로 변환해주었다.
    
    이 함수의 예를 들자면, T=[7,2]일 때 one_hot_label은
    [[0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
     [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]] 가 나오게 된다.
    '''
    
    return one_hot_label


# In[ ]:


def Softmax(ScoreMatrix): # 제공.

    if ScoreMatrix.ndim == 2:
        temp = ScoreMatrix
        temp = temp - np.max(temp, axis=1, keepdims=True)
        y_predict = np.exp(temp) / np.sum(np.exp(temp), axis=1, keepdims=True)
        return y_predict
    temp = ScoreMatrix - np.max(ScoreMatrix, axis=0)
    expX = np.exp(temp)
    y_predict = expX / np.sum(expX)
    return y_predict


# In[ ]:


def setParam_He(neuronlist):
    
    np.random.seed(1) # seed값 고정을 통해 input이 같으면 언제나 같은 Weight와 bias를 출력하기 위한 함수
    
    '''
    input layer neuron, hidden layer1 neuron, hidden layer2 neuron과 연산을 할 각각의 Weight가 필요하다.
    이 Weight값과 Bias 값을 He의 방법으로 초기화를 한다.
    He의 방법을 이용한 Weight의 초기값이란 앞 계층의 노드가 n 개일 때, 표준편차가 (2/n)^0.5 인 정규분포를 사용하는 것을 말한다.
    input layer neuron의 크기는 [데이터 개수, 784],
    hidden layer1 neuron의 크기는 [데이터 개수, 60],
    hidden layer2 neuron의 크기는 [데이터 개수, 30],
    output layer neuron의 크기는 [데이터 개수, 10] 이므로
    필요한 Weight의 크기는 순서대로 [784, 60], [60, 30], [30, 10]이 될 것이다.
    '''
    
    W1 = np.random.randn(neuronlist[0], neuronlist[1]) / np.sqrt(neuronlist[0]/2)
    W2 = np.random.randn(neuronlist[1], neuronlist[2]) / np.sqrt(neuronlist[1]/2)
    W3 = np.random.randn(neuronlist[2], neuronlist[3]) / np.sqrt(neuronlist[2]/2)
    b1 = np.zeros(neuronlist[1])
    b2 = np.zeros(neuronlist[2])
    b3 = np.zeros(neuronlist[3])
        
    return W1, W2, W3, b1, b2, b3


# In[ ]:


class linearLayer:
    '''
    이 클래스의 인스턴스는 ThreeLayerNet 클래스의 __init__() 메서드에서 만들어진다.
    '''
    def __init__(self, W, b):
        #backward에 필요한 X, W, b 값 저장 + dW, db값 받아오기
        
        self.X = None
        self.W = W
        self.b = b
        self.dW = None
        self.db = None
        
        
    def forward(self, x):
        self.X = x
        #내적연산을 통한 Z값 계산
        Z = np.dot(x, self.W) + self.b
        
        return Z
    
    def backward(self, dZ):
        #백워드 함수
        '''
        dx의 크기는 [데이터 개수, 뉴런 개수]x[뉴런 개수, 784] = [데이터 개수, 784]
        dW의 크기는 [784, 데이터 개수]x[데이터 개수, 뉴런 개수] = [784, 뉴런 개수]
        db의 크기는 [뉴런 개수, ] 이다.
        '''
        dx = np.dot(dZ, self.W.T)
        self.dW = np.dot(self.X.T, dZ)
        self.db = np.sum(dZ, axis = 0)
        
        return dx


# In[ ]:


class SiLU:
    '''
    SiLU란 f(z) = z ∗ sigmoid(z)로 나타나는 그래프이다.
    forward함수에서는, z라는 입력이 들어오면 SiLU를 activation function으로 하여 activate한 후 그 결과를 self.Z에 저장한다.
    backward함수에서는, 저장한 Z값으로 SiLU의 미분값을 구한 후 앞의 레이어에서 backward로 들어온 dActivation 값을 곱한 값 dZ를 출력한다.
    '''
    
    def __init__(self):
        self.Z = None # 백워드 시 사용할 로컬 변수
       
    
    def forward(self, Z):
        #수식에 따른 forward 함수 작성
        sig = 1 / (1 + np.exp(-Z))
        Activation = Z * sig
        self.Z = Z

        return Activation
    
    
    def backward(self, dActivation):
        '''
        연산 과정을 도식화하면 아래와 같다.
          Z                    Activation
        ----------> (SiLU) ---------------->
          dZ                 dActivation

        f'(z) = f(z) + sigmoid(z)(1-f(z)) 이다.
        '''
        sig = 1 / (1 + np.exp(-self.Z))
        fz = self.forward(self.Z)
        dZ = (sig * (1 - fz) + fz) * dActivation
        
        return dZ


# In[ ]:


class SoftmaxWithLoss(): # 제공
    
    def __init__(self):
        self.loss = None
        self.softmaxScore = None
        self.label = None
        
    def forward(self, score, one_hot_label):
        
        batch_size = one_hot_label.shape[0]
        self.label = one_hot_label
        self.softmaxScore = Softmax(score)
        self.loss = -np.sum(self.label * np.log(self.softmaxScore + 1e-20)) / batch_size
        
        return self.loss
    
    def backward(self, dout=1):
        batch_size = self.label.shape[0]
        dx = (self.softmaxScore - self.label) / batch_size
        
        return dx
                                      


# In[ ]:


class ThreeLayerNet :
    
    def __init__(self, paramlist):
        
        W1, W2, W3, b1, b2, b3 = setParam_He(paramlist)
        self.params = {}
        self.params['W1'] = W1
        self.params['W2'] = W2
        self.params['W3'] = W3
        self.params['b1'] = b1
        self.params['b2'] = b2
        self.params['b3'] = b3
        

        self.layers = OrderedDict()
        
        self.layers['L1'] = linearLayer(self.params['W1'], self.params['b1'])
        self.layers['SiLU1'] = SiLU()
        self.layers['L2'] = linearLayer(self.params['W2'], self.params['b2'])
        self.layers['SiLU2'] = SiLU()
        self.layers['L3'] = linearLayer(self.params['W3'], self.params['b3'])
        
        self.lastLayer = SoftmaxWithLoss()
        
    def scoreFunction(self, x):
        
        for layer in self.layers.values():
            # 한 줄이 best
            x = layer.forward(x)
        
        score = x
        return score
        
    def forward(self, x, label):
        
        score = self.scoreFunction(x)
        return self.lastLayer.forward(score, label)
    
    def accuracy(self, x, label):
        
        score = self.scoreFunction(x)
        score_argmax = np.argmax(score, axis=1)
        
        if label.ndim != 1 : #label이 one_hot_encoding 된 데이터면 if문을 
            label_argmax = np.argmax(label, axis = 1)
            
        accuracy = np.sum(score_argmax==label_argmax) / int(x.shape[0])
        
        return accuracy
    
    def backpropagation(self):
        
        #백워드 함수 작성 스코어펑션을 참고하세요
        dL = self.lastLayer.backward()
        dA2 = self.layers['L3'].backward(dL)
        dZ2 = self.layers['SiLU2'].backward(dA2)
        dA1 = self.layers['L2'].backward(dZ2)
        dZ1 = self.layers['SiLU1'].backward(dA1)
        d = self.layers['L1'].backward(dZ1)
            
        grads = {}
        grads['W1'] = self.layers['L1'].dW
        grads['b1'] = self.layers['L1'].db
        grads['W2'] = self.layers['L2'].dW
        grads['b2'] = self.layers['L2'].db
        grads['W3'] = self.layers['L3'].dW
        grads['b3'] = self.layers['L3'].db
        
        return grads
    
    def gradientdescent(self, grads, learning_rate):
        
        self.params['W1'] -= learning_rate*grads['W1']
        self.params['W2'] -= learning_rate*grads['W2']
        self.params['W3'] -= learning_rate*grads['W3']
        self.params['b1'] -= learning_rate*grads['b1']
        self.params['b2'] -= learning_rate*grads['b2']
        self.params['b3'] -= learning_rate*grads['b3']


    def forward_for_dropout(self, x, label, kill_1, kill_2):
        x = self.layers['L1'].forward(x)
        x = self.layers['SiLU1'].forward(x)
        u1 = np.random.rand(*x.shape) > kill_1
        x *= u1
        x = self.layers['L2'].forward(x)
        x = self.layers['SiLU2'].forward(x)
        u2 = np.random.rand(*x.shape) > kill_2
        x *= u2
        x = self.layers['L3'].forward(x)
        loss = self.lastLayer.forward(x, label)

        return loss


# In[ ]:


def batchOptimization(dataset, ThreeLayerNet, learning_rate, epoch=1000):
    train_acc_list = []
    test_acc_list = []
    Loss_list = []

    for i in range(epoch+1):
        #코드 작성
        Loss = ThreeLayerNet.forward(dataset['train_X'], dataset['one_hot_train'])
        ThreeLayerNet.gradientdescent(ThreeLayerNet.backpropagation(), learning_rate)
        
        if i % 10 == 0:
            train_acc = ThreeLayerNet.accuracy(dataset['train_X'], dataset['one_hot_train'])
            test_acc = ThreeLayerNet.accuracy(dataset['test_X'], dataset['one_hot_test'])
            print(i, '\t번째 Loss = ', Loss)
            print(i, '\t번째 Train_Accuracy : ', train_acc)
            print(i, '\t번째 Test_Accuracy : ', test_acc)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            Loss_list.append(Loss)
   
    return ThreeLayerNet, train_acc_list, test_acc_list, Loss_list


# In[ ]:


def minibatch_Optimization(dataset, ThreeLayerNet, learning_rate, epoch=100, batch_size=1000):
    '''

    '''
    train_acc_list = []
    test_acc_list = []
    Loss_list = []

    np.random.seed(5)
    for i in range(epoch+1):
        # 코드 작성

        tmp_train = np.concatenate((dataset['one_hot_train'], dataset['train_X']), axis=1)
        np.random.shuffle(tmp_train)
        dataset['one_hot_train'] = tmp_train[:, :10]
        dataset['train_X'] = tmp_train[:, 10:]

        batch = {}
        b = 0
        while b < (tmp_train.shape[0] / batch_size):
            batch['train_X'] = dataset['train_X'][(batch_size * b):(batch_size * (b+1)), :]
            batch['one_hot_train'] = dataset['one_hot_train'][(batch_size * b):(batch_size * (b+1)), :]
            Loss = ThreeLayerNet.forward(batch['train_X'], batch['one_hot_train'])
            ThreeLayerNet.gradientdescent(ThreeLayerNet.backpropagation(), learning_rate)
            b += 1
        
        if i % 10 == 0:
            train_acc = ThreeLayerNet.accuracy(dataset['train_X'], dataset['one_hot_train'])
            test_acc = ThreeLayerNet.accuracy(dataset['test_X'], dataset['one_hot_test'])
            print(i, '\t번째 Loss = ', Loss)
            print(i, '\t번째 Train_Accuracy : ', train_acc)
            print(i, '\t번째 Test_Accuracy : ', test_acc)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            Loss_list.append(Loss)  

    return ThreeLayerNet, train_acc_list, test_acc_list, Loss_list


# In[ ]:


def dropout_use_Optimizer(dataset, ThreeLayerNet, learning_rate, epoch, kill_n_h1 = 0.25, kill_n_h2 = 0.15):
    train_acc_list = []
    test_acc_list = []
    Loss_list = []

    for i in range(epoch+1):
        #코드 작성
        Loss = ThreeLayerNet.forward_for_dropout(dataset['train_X'], dataset['one_hot_train'], kill_n_h1, kill_n_h2)
        ThreeLayerNet.gradientdescent(ThreeLayerNet.backpropagation(), learning_rate)
        
        if i % 10 == 0:
            train_acc = ThreeLayerNet.accuracy(dataset['train_X'], dataset['one_hot_train'])
            test_acc = ThreeLayerNet.accuracy(dataset['test_X'], dataset['one_hot_test'])
            print(i, '\t번째 Loss = ', Loss)
            print(i, '\t번째 Train_Accuracy : ', train_acc)
            print(i, '\t번째 Test_Accuracy : ', test_acc)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            Loss_list.append(Loss)  
    return ThreeLayerNet, train_acc_list, test_acc_list, Loss_list


# In[ ]:


#과제 채점을 위한 세팅
train_X, train_label, test_X, test_label = train_test_split(mnist)

one_hot_train = one_hot_encoding(train_label)
one_hot_test = one_hot_encoding(test_label)

dataset = {}
dataset['train_X'] = train_X
dataset['test_X'] = test_X
dataset['one_hot_train'] = one_hot_train
dataset['one_hot_test'] = one_hot_test

neournlist = [784, 60, 30, 10]

TNN_batchOptimizer = ThreeLayerNet(neournlist)
TNN_minibatchOptimizer = copy.deepcopy(TNN_batchOptimizer)
TNN_dropOut = copy.deepcopy(TNN_minibatchOptimizer)


# In[ ]:


#채점은 이 것의 결과값으로 할 예정입니다. 

#trained_batch, tb_train_acc_list, tb_test_acc_list, tb_loss_list = batchOptimization(dataset, TNN_batchOptimizer, 0.1, 500)
trained_minibatch, tmb_train_acc_list, tmb_test_acc_list, tb_loss_list = minibatch_Optimization(dataset, TNN_minibatchOptimizer, 0.01, epoch=100, batch_size=1000)
#trained_dropout, td_train_acc_list, td_test_acc_list, td_loss_list = dropout_use_Optimizer(dataset, TNN_dropOut, 0.1, 1000, 0.25, 0.15)

