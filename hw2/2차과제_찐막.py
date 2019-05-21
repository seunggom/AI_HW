#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from collections import OrderedDict
import copy


# In[ ]:


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


# In[ ]:


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
    그리고 one_hot_label에서 각 행마다 정답 인덱스에 해당하는 열에 1의 값을 저장한다.
    
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
    bias의 경우에는 초기값으로 전부 0의 값을 제공한다.
    bias의 크기는 순서대로 [60, ], [30, ], [10, ] 이다.
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
    각각의 레이어에 필요한 forward, backward 연산을 하는 함수를 제공한다.
    '''
    def __init__(self, W, b):
        #backward에 필요한 X, W, b 값 저장 + dW, db값 받아오기
        
        self.X = None
        self.W = W
        self.b = b
        self.dW = None
        self.db = None
        
        
    def forward(self, x):
        '''
        내적연산을 통한 Z값 계산하는 함수이다.
        Z가 의미하는 것은 이전 레이어의 뉴런들에 weight만큼의 가중치를 적용한 신호의 총합들이다.
        즉 값이 높은 뉴런일 수록 그것이 정답일 가능성이 높다고 추측한 것이다.
        '''
        self.X = x
        Z = np.dot(x, self.W) + self.b
        
        return Z
    
    
    def backward(self, dZ):
        #백워드 함수
        '''
        gradient를 계산하는 함수이다.
        dx의 크기는 [데이터 개수, 뉴런 개수]x[뉴런 개수, 784] = [데이터 개수, 784]
        dW의 크기는 [784, 데이터 개수]x[데이터 개수, 뉴런 개수] = [784, 뉴런 개수]
        db의 크기는 [뉴런 개수, ] 이므로,
        크기에 맞게 내적연산을 진행한다.
        '''
        dx = np.dot(dZ, self.W.T)
        self.dW = np.dot(self.X.T, dZ)
        self.db = np.sum(dZ, axis = 0)
        
        return dx


# In[ ]:


class Dropout:
    '''
    드랍아웃을 위한 클래스이다.
    드랍아웃을 사용할 경우 이 클래스의 객체가 생성되고, 각 히든 레이어마다 kill_rate만큼 값을 죽인다.
    '''
    def __init__(self):
        self.kill_rate = None
        self.U = None

    def forward(self, D, train_flag=True):
        '''
        드랍아웃으로 뉴런을 죽이는 것은 training 과정에서만 진행해야하므로
        forward는 train_flag로 training과 testing 과정을 구분하여 진행한다.
        U = np.random.rand(*D.shape) >= kill_rate 의 의미를 설명하자면,
        np.random.rand(*D.shape)은 D의 크기만큼 0에서 1사이의 값을 가지는 행렬을 리턴한다.
        이 때 kill_rate 이상인 값을 가지면 식이 참이므로 1의 값을 가지고 kill_rate 미만의 값을 가지면 식이 거짓이므로 0을 가진다.
        그래서 0 혹은 1의 값을 가지고 있는 U을 x에 곱하면, kill_rate의 확률만큼 뉴런이 죽게된다.
        
        testing 과정에서는 모든 뉴런을 사용하기 때문에
        training 과정에서의 출력 뉴런 데이터의 기대값과 동일한 기대값을 갖기 위해서는 (1 - kill_rate) 만큼 곱해주어야 한다.
        '''
        if train_flag:
            self.U = np.random.rand(*D.shape) >= self.kill_rate
            return D * self.U
        else:
            return D * (1 - self.kill_rate)

    def backward(self, dout):
        '''
        forward 과정에서 죽였던 뉴런 위치 그대로 backward를 진행한다.
        '''
        dx = dout * self.U
        return dx


# In[ ]:


class SiLU:
    '''
    SiLU란 f(z) = z ∗ sigmoid(z)로 나타나는 함수이다.
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

        이 때, f'(z) = f(z) + sigmoid(z)(1-f(z)) 이므로
        dZ = (f(z) + sigmoid(z)(1-f(z))) * dActivation 이다.
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
        '''
        모든 레이어에 대해 차례대로 forward를 진행한다.
        그리고 리턴값으로는 forward를 모두 마친 후의 score값을 리턴한다.
        '''
        for layer in self.layers.values():
            # 한 줄이 best
            x = layer.forward(x)
        
        score = x
        return score
        
    def forward(self, x, label):
        '''
        위의 scoreFunction 함수를 이용해 score를 구하고,
        loss를 구하여 리턴하는 함수이다.
        '''
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
        '''
        forward를 진행한 레이어 순서의 반대 순서로 backward를 진행한다.
        backward 후, 각 레이어 내 저장되어 있는 dW, db값을 grads 라는 딕셔너리 객체에 저장한 후 그 값을 리턴한다.
        
        '''
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


# In[ ]:


class UseDropout_ThreeLayerNet:
    '''
    처음엔 기존의 ThreeLayerNet 클래스를 수정하여 드랍아웃 구현을 하려고 했는데,
    구현할수록 코드가 복잡해져서 드랍아웃 과정이 추가된 ThreeLayerNet 클래스를 따로 구현하였다.
    '''
    
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

        #드랍아웃을 위한 레이어 두개
        self.dropout_layer1 = Dropout()
        self.dropout_layer2 = Dropout()
        
        self.lastLayer = SoftmaxWithLoss()
        
        
    def scoreFunction(self, x, train_flag):
        '''
        레이어 순서대로 forward를 진행하는데, 뉴런값들을 activate한 후에 train_flag에 따라서 dropout을 진행한다.
        '''
        x = self.layers['L1'].forward(x)
        x = self.layers['SiLU1'].forward(x)
        u1 = self.dropout_layer1.forward(x, train_flag)
        x = self.layers['L2'].forward(u1)
        x = self.layers['SiLU2'].forward(x)
        u2 = self.dropout_layer2.forward(x, train_flag)
        x = self.layers['L3'].forward(u2)
        
        score = x
        return score
        
    def forward(self, x, label, train_flag=True):
        '''
        score를 계산하고, 그것에 따라 Loss를 리턴한다.
        '''
        score = self.scoreFunction(x, train_flag)

        return self.lastLayer.forward(score, label)
    
    def accuracy(self, x, label):
        '''
        정확도를 구하는 함수이다.
        이 때는 오로지 각 데이터셋에 대한 정확도를 구하면 되기 때문에 train_flag를 false로 설정하여 뉴런을 죽이는 과정은 하지 않는다.
        '''
        train_flag = False
        score = self.scoreFunction(x, train_flag)
        score_argmax = np.argmax(score, axis=1)
        
        if label.ndim != 1 : #label이 one_hot_encoding 된 데이터면 if문을 
            label_argmax = np.argmax(label, axis = 1)
            
        accuracy = np.sum(score_argmax==label_argmax) / int(x.shape[0])
        
        return accuracy
    
    def backward(self):
        '''
        dropout 레이어의 forward에서 지정한 확률만큼 뉴런을 죽인것처럼, backward에서도 똑같이 gradient값을 죽인다.
        forward에서 진행한 레이어 순서의 반대로 진행한다.
        '''
        dL = self.lastLayer.backward()
        dA2 = self.layers['L3'].backward(dL)
        dD2 = self.dropout_layer2.backward(dA2)
        dZ2 = self.layers['SiLU2'].backward(dD2)
        dA1 = self.layers['L2'].backward(dZ2)
        dD1 = self.dropout_layer1.backward(dA1)
        dZ1 = self.layers['SiLU1'].backward(dD1)
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
        
        
    def setKillRate(self, r1, r2):
        '''
        각 히든레이어에서 쓸 kill_rate의 값을 받아서 업데이트한다.
        '''
        self.dropout_layer1.kill_rate = r1
        self.dropout_layer2.kill_rate = r2


# In[ ]:


def batchOptimization(dataset, ThreeLayerNet, learning_rate, epoch=1000):
    '''
    이 함수에서는 한 epoch마다 8000개의 train_X를 한 번에 forward 해서 Loss를 구하고,
    backpropagation, gradientdescent를 사용해 W와 b를 업데이트 한다.
    그리고 10번째마다 loss와 정확도를 출력한다.
    '''
    
    for i in range(epoch+1):
        #코드 작성
        train_acc_list = []
        test_acc_list = []
        Loss_list = []
        # 위의 것들은 append를 할 때 필요한 것이다. append()는 기존 리스트의 뒤에 새로 추가하라는 의미이다.
        
        Loss = ThreeLayerNet.forward(dataset['train_X'], dataset['one_hot_train'])
        grad = ThreeLayerNet.backpropagation()
        ThreeLayerNet.gradientdescent(grad, learning_rate)
        
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
    이 함수에서는 minibatch로 나누어서 optimization을 진행한다.
    먼저 train_X와 one_hot_train을 random하게 섞는데, 두 배열 간의 관계가 달라지면 안되기 때문에
    np.concatenate()를 이용해 임시로 두 배열을 합치고 나서 np.random.shuffle()을 이용해 랜덤하게 섞는다.
    그러고나서 다시 처음처럼 train_X와 one_hot_train을 분할한다.
    한 epoch는 모든 minibatch들을 다 진행했을 때를 말한다.
    그래서 미니배치 사이즈가 100이고 데이터의 갯수가 8000개라면 총 80번의 minibatch를 돌아야 1 epoch인 것이다.
    '''

    np.random.seed(5)
    for i in range(epoch+1):
        # 코드 작성
        train_acc_list = []
        test_acc_list = []
        Loss_list = []

        tmp_train = np.concatenate((dataset['one_hot_train'], dataset['train_X']), axis=1)
        np.random.shuffle(tmp_train)
        dataset['one_hot_train'] = tmp_train[:, :10]
        dataset['train_X'] = tmp_train[:, 10:]

        batch = {}
        b = 0
        '''
        미니배치 크기가 100이고 데이터의 갯수가 8000개인 경우
        위에서 랜덤하게 섞은 데이터를 매 반복문마다 슬라이싱을 이용해
        0~99번째, 100~199번째, ..., 7900~7999번째 데이터로 나누어서 연산을 진행한다.
        '''
        while b < (tmp_train.shape[0] / batch_size):
            batch['train_X'] = dataset['train_X'][(batch_size * b):(batch_size * (b+1)), :]
            batch['one_hot_train'] = dataset['one_hot_train'][(batch_size * b):(batch_size * (b+1)), :]
            
            Loss = ThreeLayerNet.forward(batch['train_X'], batch['one_hot_train'])
            grad = ThreeLayerNet.backpropagation()
            ThreeLayerNet.gradientdescent(grad, learning_rate)
            # 각각의 미니배치마다 loss, gradient를 구해서 weight와 bias를 업데이트 한다.
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


def dropout_use_Optimizer(dataset, UseDropout_ThreeLayerNet, learning_rate, epoch, kill_n_h1 = 0.25, kill_n_h2 = 0.15):
    '''
    Dropout을 사용한 Optimization이다.
    Dropout을 사용하는 이유는 train data에만 치중하여 weight를 업데이트하면
    오히려 test data의 정확도가 떨어질 수 있어 이를 방지하기 위함이다.
    kill_n_h1, kill_n_h2은 드롭아웃으로 뉴런을 죽이는 비율이다.
    UseDropout_ThreeLayerNet 클래스를 이용하였다.
    '''
    
    UseDropout_ThreeLayerNet.setKillRate(kill_n_h1, kill_n_h2) # 뉴런을 죽일 비율을 dropout 레이어에 전달한다.
    
    for i in range(epoch+1):
        #코드 작성
        train_acc_list = []
        test_acc_list = []
        Loss_list = []
    
        Loss = UseDropout_ThreeLayerNet.forward(dataset['train_X'], dataset['one_hot_train'])
        grad = UseDropout_ThreeLayerNet.backward()
        UseDropout_ThreeLayerNet.gradientdescent(grad, learning_rate)
        
        if i % 10 == 0:
            train_acc = UseDropout_ThreeLayerNet.accuracy(dataset['train_X'], dataset['one_hot_train'])
            test_acc = UseDropout_ThreeLayerNet.accuracy(dataset['test_X'], dataset['one_hot_test'])
            print(i, '\t번째 Loss = ', Loss)
            print(i, '\t번째 Train_Accuracy : ', train_acc)
            print(i, '\t번째 Test_Accuracy : ', test_acc)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            Loss_list.append(Loss)
            
    return UseDropout_ThreeLayerNet, train_acc_list, test_acc_list, Loss_list


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
TNN_dropOut = UseDropout_ThreeLayerNet(neournlist) # UseDropout_ThreeLayerNet 클래스 객체 생성함


# In[ ]:


#채점은 이 것의 결과값으로 할 예정입니다. 

trained_batch, tb_train_acc_list, tb_test_acc_list, tb_loss_list = batchOptimization(dataset, TNN_batchOptimizer, 0.1, 1000)
trained_minibatch, tmb_train_acc_list, tmb_test_acc_list, tb_loss_list = minibatch_Optimization(dataset, TNN_minibatchOptimizer, 0.1, epoch=100, batch_size=100)
trained_dropout, td_train_acc_list, td_test_acc_list, td_loss_list = dropout_use_Optimizer(dataset, TNN_dropOut, 0.1, 1000, 0.25, 0.15)

