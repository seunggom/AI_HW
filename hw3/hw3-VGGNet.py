#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!pip install cifar10_web # 리눅스 환경에선 이 부분을 지우시고 pip install cifar10_web 으로 직접 인스톨 해주세요.
import numpy as np
from collections import OrderedDict
import cifar10_web
# import matplotlib.pyplot as plt 이 부분은 맨 밑의 이미지를 확인하고 싶을 때 #을 지워서 확인해주세요.
import pickle


# In[ ]:


# train_images, train_labels, test_images, test_labels = cifar10_web.cifar10(path=None)


# In[ ]:


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """다수의 이미지를 입력받아 2차원 배열로 변환한다(평탄화).
    
    Parameters
    ----------
    input_data : 4차원 배열 형태의 입력 데이터(이미지 수, 채널 수, 높이, 너비)
    filter_h : 필터의 높이
    filter_w : 필터의 너비
    stride : 스트라이드
    pad : 패딩
    
    Returns
    -------
    col : 2차원 배열
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col


# In[ ]:


def load_params(ShallowCNN, file_name="params3rd.pkl"):
    with open(file_name, 'rb') as f:
        values = pickle.load(f)
  
    for key, val in ShallowCNN.layers.items():
        if key in values.keys():
            W, b = values[key]
            ShallowCNN.layers[key].W = W
            ShallowCNN.layers[key].b = b      
    
  


# In[ ]:


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """(im2col과 반대) 2차원 배열을 입력받아 다수의 이미지 묶음으로 변환한다.
    
    Parameters
    ----------
    col : 2차원 배열(입력 데이터)
    input_shape : 원래 이미지 데이터의 형상（예：(10, 1, 28, 28)）
    filter_h : 필터의 높이
    filter_w : 필터의 너비
    stride : 스트라이드
    pad : 패딩
    
    Returns
    -------
    img : 변환된 이미지들
    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]


# In[ ]:


class LinearLayer:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.dW = None
        self.db = None
        self.X = None
    
    def forward(self, x):
        self.OrX = x.shape
        x = x.reshape(x.shape[0], -1)
            
        self.X = x

        Z = np.dot(self.X, self.W) + self.b
        return Z
        
    def backward(self, dout):
        self.dW = np.dot(self.X.T, dout)
        self.db = np.sum(dout, axis=0)
        dx = np.dot(dout, self.W.T)
        
        dx = dx.reshape(*self.OrX)
    
        return dx


# In[ ]:


class Convolution:
    
    def __init__(self, W, b, stride=1, pad=0):
        
        ''' 
        Convolution Layer i의 모든 필터의 Weight를 저장. shape[0]은 필터의 개수 shape[1]은 Channel의 개수 
        Shape[2]는 필터의 높이 shape[3]은 필터의 가로
        '''
        
        self.W = W
        
        '''
        bias는 각 filter마다 1개만 있으면 되기에 (FN, 1, 1, 1)의 shape를 가진다.
        '''

        self.b = b
   
        self.stride = stride
        self.pad = pad
        
    def forward(self, x):
        '''
        FN, C, FH, FW에 Filter W의 shape를 저장
        '''
        (FN, C, FH, FW) = self.W.shape
        '''
        input data x 또한 4차원의 데이터이다. N은 데이터의 개수 C는 채널의 개수 H, W 는 Height, Width
        
        '''
        (N, C, H, W) = x.shape
        
        out_h = int(1+ (H+2*self.pad - FH) / self.stride)
        out_W = int(1+ (W+2*self.pad - FW) / self.stride)
        
        ''' 
        데이터를 2차원으로 바꾸어서 np.dot 연산으로 한 번에 연산을 가능하게 함
        '''
        col = im2col(x, FH, FW, self.stride, self.pad) 
    
        ''' 
        필터를 2차원으로 바꾸어서 np.dot 연산으로 한 번에 연산을 가능하게 함
        '''
        col_W = self.W.reshape(FN, -1).T #필터의 전개
        out = np.dot(col, col_W)
        out = out.reshape(N, out_h, out_W, -1).transpose(0, 3, 1, 2)
        
        self.x = x
        self.col = col
        self.col_W = col_W
        
        return out
    
    
    def backward(self, dout):
        
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)
        
        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)
        
        return dx

        


# In[ ]:


#Convolution Layer 사용 예시
#filter개수 4개, filter channel 3 filter H 
W = np.random.randn(4, 3, 3, 3) 
b = np.random.randn(2, 1, 1, 1)
con1 = Convolution(W, b, stride=1, pad=1)


# In[ ]:


class Pooling:
    def __init__(self, pool_size, stride=1, pad=0):
        self.pool_h = pool_size
        self.pool_w = pool_size
        self.stride = stride
        self.pad = pad
        
        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max
        

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx


# In[ ]:


class ReLU:
    def __init__(self):
        self.Z = None
        
    
    def forward(self, Z):
        self.Z = Z
        self.mask = (Z<0)
        A = Z.copy()
        A[self.mask] = 0
        
        return A
    
    def backward(self, dout):
        dA = dout
        dA[self.mask] = 0
        
        return dA
    


# In[ ]:


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))


# In[ ]:


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None # 손실함수
        self.y = None    # softmax의 출력
        self.t = None    # 정답 레이블(원-핫 인코딩 형태)
        
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = -np.sum(self.t * np.log(self.y + 1e-6))/ x.shape[0]
        
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 정답 레이블이 원-핫 인코딩 형태일 때
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx


# In[ ]:


def filterParamSet(filter_num , filter_size, filter_channel, prev_filter_num=1):
    #그 필터에 해당하는 Weight와 bias 생성 간단하게 0.01로 초기화
    filterWeight = np.random.randn(filter_num, filter_channel, filter_size, filter_size) * np.sqrt(2/(filter_size*filter_size*filter_num))
    filterbias = np.zeros(filter_num)
    
    return filterWeight, filterbias 


# In[ ]:


def fullLayerParamSet(input_n, output_n):
    #해당 fullLayer에 해당하는 Weight와 bias 생성
    full_W = np.random.randn(input_n, output_n) * np.sqrt(2/input_n)
    full_b = np.zeros([1, output_n])
    return full_W, full_b


# In[ ]:


def Convolution_Layers_set(ConvLayerlist):
    Convolution_Layers = OrderedDict()
    """
    ConvLayerList[0] = Convolution이면 'C' Pooling이면 'P'
    ConvLayerList[1] = Convolution이면 'filter의 개수(filter_num)' Pooling이면 'filter_size'
    ConvLayerList[2] = Convolution이면 'filter의 크기(filter_size)' Pooling이면 'stride'
    ConvLayerList[3] = Convolution이면 'filter의 채널(filter_channel)' Pooling이면 'pad'
    ConvLayerList[4] = Convolution이면 'stride'
    ConvLayerList[5] = Convolution이면 'pad'
    """

    for i in range(len(ConvLayerlist)):
        if ConvLayerlist[i][0] == 'C':
            C_Weight, C_bias = filterParamSet(ConvLayerlist[i][1], ConvLayerlist[i][2], ConvLayerlist[i][3])
            Convolution_Layers['C'+str(i+1)] = Convolution(C_Weight, C_bias, ConvLayerlist[i][4], ConvLayerlist[i][5])
        elif ConvLayerlist[i][0] == 'P':
            Convolution_Layers['P'+str(i+1)] = Pooling(ConvLayerlist[i][1], ConvLayerlist[i][2], ConvLayerlist[i][3])
            
    return Convolution_Layers


# In[ ]:


def FullyConnected_Layers_set(FullyConnectedLayerlist):
    FullyConnected_layers = OrderedDict()

    for i in range(len(FullyConnectedLayerlist)):
        full_W, full_b = fullLayerParamSet(FullyConnectedLayerlist[i][0], FullyConnectedLayerlist[i][1])
        FullyConnected_layers['F'+str(i+1)] = LinearLayer(full_W, full_b)
    
    return FullyConnected_layers


# In[ ]:


class VGGNet:
    
    def __init__(self, ConvLayerlist, FullLayerlist):
        np.random.seed(1)
        self.Convolution_Layers = Convolution_Layers_set(ConvLayerlist)
        self.FC_Layers = FullyConnected_Layers_set(FullLayerlist)
        
        self.layers = OrderedDict()
        self.i = 0
        
        for layer in self.Convolution_Layers.values():
            self.i = self.i+1
            if(type(layer) ==Convolution):
                self.layers['C'+str(self.i)] = layer
                self.layers['R'+str(self.i)] = ReLU()
            elif(type(layer)==Pooling):
                self.layers['P'+str(self.i)] = layer
            else:
                print("이상한게 들어왔네요")
            
        for layer in self.FC_Layers.values():
            self.i = self.i+1
            self.layers['F'+str(self.i)] = layer
            self.layers['R'+str(self.i)] = ReLU()
            
        """
        이 부분을 수정해야 할 것
        """
        
        '''
        바로 이전 fully-connected layer의 크기는 (4096, )이고, cifar10의 레이어 개수는 10개이므로,
        마지막 fully-connected layer 계산에 필요한 weight 크기는 (4096, 10)이어야 한다.
        '''
        last_f_w, last_f_b = np.random.randn(4096, 10)*0.01, np.zeros([1,10])
        self.i = self.i+1
        self.layers['F'+str(self.i)] = LinearLayer(last_f_w, last_f_b)
        self.lastlayer = SoftmaxWithLoss()
        
        
    #Score를 구하는 함수    
    def Score(self, x):
        self.x = x
        for layer in self.layers.values():
            x = layer.forward(x)
            
        return x
    
    #Loss를 구하는 함수
    def forward(self, x, t):
        y = self.Score(x)
        loss = self.lastlayer.forward(y, t)
        return loss
    
        
    #구현하세요.

        

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        acc = 0.0
        
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.Score(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt) 
        
        return acc / x.shape[0]
      
      


# In[ ]:


class VGGNet16:
    
    def __init__(self, ConvLayerlist, FullLayerlist):
        np.random.seed(1)
        self.Convolution_Layers = Convolution_Layers_set(ConvLayerlist)
        self.FC_Layers = FullyConnected_Layers_set(FullLayerlist)
        
        self.layers = OrderedDict()
        self.i = 0
        
        for layer in self.Convolution_Layers.values():
            self.i = self.i+1
            if(type(layer) ==Convolution):
                self.layers['C'+str(self.i)] = layer
                self.layers['R'+str(self.i)] = ReLU()
            elif(type(layer)==Pooling):
                self.layers['P'+str(self.i)] = layer
            else:
                print("이상한게 들어왔네요")
            
        for layer in self.FC_Layers.values():
            self.i = self.i+1
            self.layers['F'+str(self.i)] = layer
            self.layers['R'+str(self.i)] = ReLU()
            
        """
        이 부분 또한 수정해야 할 것
        """
        
        '''
        바로 이전 fully-connected layer의 크기는 (4096, )이고, cifar10의 레이어 개수는 10개이므로,
        마지막 fully-connected layer 계산에 필요한 weight 크기는 (4096, 10)이어야 한다.
        '''
        last_f_w, last_f_b = np.random.randn(4096, 10)*0.01, np.zeros([1,10])
        self.i = self.i+1
        self.layers['F'+str(self.i)] = LinearLayer(last_f_w, last_f_b)
        self.lastlayer = SoftmaxWithLoss()
        
        
    #Score를 구하는 함수    
    def Score(self, x):
        self.x = x
        for layer in self.layers.values():
            x = layer.forward(x)
            
        return x
    
    #Loss를 구하는 함수
    def forward(self, x, t):
        y = self.Score(x)
        loss = self.lastlayer.forward(y, t)
        return loss
    
        

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        acc = 0.0
        
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.Score(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt) 
        
        return acc / x.shape[0]
      
      


# In[ ]:


"""
VGG는 항상 3x3 크기의 필터에 stride=1, pad=1를 사용한다.
인풋 데이터의 크기가 32x32x3일 때 VGG11의 convolution 결과를 나열하면,

32x32x64 -> 16x16x64
-> 16x16x128 -> 8x8x128
-> 8x8x256 -> 8x8x256 ->4x4x256
-> 4x4x512 -> 4x4x512 -> 2x2x512
-> 2x2x512 -> 2x2x512 -> 1x1x512 이다.

fully connected layer 경우에는 FC-4096, FC-4096에 의해
(512, 4096), (4096,4096)의 weight 크기가 필요하다.
"""

VGG11ConvLayerlist = [['C', 64, 3, 3, 1, 1], ['P', 2, 2, 0],
                      ['C', 128, 3, 64, 1, 1], ['P', 2, 2, 0],
                      ['C', 256, 3, 128, 1, 1], ['C', 256, 3, 256, 1, 1], ['P', 2, 2, 0],
                      ['C', 512, 3, 256, 1, 1], ['C', 512, 3, 512, 1, 1], ['P', 2, 2, 0],
                      ['C', 512, 3, 512, 1, 1], ['C', 512, 3, 512, 1, 1], ['P', 2, 2, 0]]
VGG11FullLayerlist = [[1*1*512, 4096], [4096, 4096]]

VGGNet11 = VGGNet(VGG11ConvLayerlist, VGG11FullLayerlist)


# In[ ]:


"""
VGG16의 경우에는 1x1 filter를 쓰는 쓰는 단계가 있고, 그 외 맥락은 위와 비슷하다.
VGG16의 convolution 결과를 나열하면,

32x32x64 -> 32x32x64 -> 16x16x64
-> 16x16x128 -> 16x16x128 -> 8x8x128
-> 8x8x256 -> 8x8x256 -> 10x10x256 -> 5x5x256
-> 5x5x512 -> 5x5x512 -> 7x7x512 -> 3x3x512
-> 3x3x512 -> 3x3x512 -> 5x5x512 -> 2x2x512 이다.

그래서 fully connected layer 경우에는 FC-4096, FC-4096에 의해
(2x2x512, 4096), (4096,4096)의 weight 크기가 필요하다.

"""

VGG16ConvLayerlist = [['C', 64, 3, 3, 1, 1], ['C', 64, 3, 64, 1, 1], ['P', 2, 2, 0],
                      ['C', 128, 3, 64, 1, 1], ['C', 128, 3, 128, 1, 1], ['P', 2, 2, 0],
                      ['C', 256, 3, 128, 1, 1], ['C', 256, 3, 256, 1, 1],['C', 256, 1, 256, 1, 1], ['P', 2, 2, 0],
                      ['C', 512, 3, 256, 1, 1], ['C', 512, 3, 512, 1, 1],['C', 512, 1, 512, 1, 1], ['P', 2, 2, 0],
                      ['C', 512, 3, 512, 1, 1], ['C', 512, 3, 512, 1, 1],['C', 512, 1, 512, 1, 1], ['P', 2, 2, 0]]
VGG16FullLayerlist = [[2*2*512, 4096], [4096, 4096]]

VGGNet16 = VGGNet16(VGG16ConvLayerlist, VGG16FullLayerlist)


# In[ ]:


print("VGG11에 대한 내용")
print("전체적인 Layer", VGGNet11.layers.keys())
print("---------------------------------------------------------")
for key, layer in VGGNet11.layers.items():
    
    if type(layer) == Convolution:
        print("이 레이어는 ", key, "이고, W의 Shape는 ", layer.W.shape, "이다")
    elif type(layer) == LinearLayer:
        print("이 레이어는 ",key, "이고, W의 Shape는 ", layer.W.shape, "이다")


# In[ ]:


print("VGG16에 대한 내용")
print("전체적인 Layer", VGGNet16.layers.keys())
print("---------------------------------------------------------")
for key, layer in VGGNet16.layers.items():
    
    if type(layer) == Convolution:
        print("이 레이어는 ", key, "이고, W의 Shape는 ", layer.W.shape, "이다")
    elif type(layer) == LinearLayer:
        print("이 레이어는 ",key, "이고, W의 Shape는 ", layer.W.shape, "이다")


# In[ ]:




