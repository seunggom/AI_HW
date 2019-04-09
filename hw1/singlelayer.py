# B635076 김승연
import numpy as np


class singleLayer:
    def __init__(self, W, Bias):  # 제공. 호출 시 작동하는 생성자
        self.W = W
        self.B = Bias

    def SetParams(self, W_params, Bias_params):  # 제공. W와 Bias를 바꾸고 싶을 때 쓰는 함수
        self.W = W_params
        self.B = Bias_params

    def ScoreFunction(self, X):  # \Score값 계산 -> 직접작성
        # 3.2
        """
        매개변수로 받은 X는 이미지 데이터임
        X * W(weight) + B(bias) 를 계산하여 결과값을 ScoreMatrix 에 저장하여 이를 리턴함
        X는 [이미지 개수, 784]인 배열, W는 [784, 10(label 개수)]인 배열이므로 np.dot() 함수를 이용하여 곱셈을 하고
        가중치인 B를 더해주면 결과로 나올 ScoreMatrix는 [이미지 개수, 10]인 배열임
        예를 들어 ScoreMatrix의 3번째 행에서 가장 큰값을 가진 열(index)이 6이라면, 3번째 이미지가 숫자 6일 것이라 추측한 것임
        """

        ScoreMatrix = np.dot(X, self.W) + self.B

        return ScoreMatrix

    def Softmax(self, ScoreMatrix):  # 제공.
        if ScoreMatrix.ndim == 2:
            temp = ScoreMatrix.T
            temp = temp - np.max(temp, axis=0)  # 가장 점수가 높은 인덱스의 값으로 뺌
            y_predict = np.exp(temp) / np.sum(np.exp(temp), axis=0)
            return y_predict.T
        temp = ScoreMatrix - np.max(ScoreMatrix, axis=0)
        expX = np.exp(temp)
        y_predict = expX / np.sum(expX)
        return y_predict # 사진마다의 y_predict의 모든 데이터의 합은 1임

    def LossFunction(self, y_predict, Y):  # Loss Function을 구하십시오 -> 직접 작성
        # 3.3
        """
        :param y_predict:
            Softmax 함수를 이용해서 얻은 결과값임.
            처음에 ScoreFuction을 이용해 ScoreMatrix를 계산했는데, 이 값들을 0~1 사이의 숫자, 즉 확률로 표현한 값임.
        :param Y: 이미지 데이터의 정답 레이블임. one-hot 배열로, 정답인 인덱스에만 1이고 나머지는 0임
        :return: 사진의 정답값과 y_predict에서 가장 큰 값이 있는 인덱스가 틀릴 경우, 이 정답을 얼마나 제대로 맞추지 못했는지를 알려주는 척도임
        즉, 리턴값인 loss가 작을 수록 잘 추측한 것임

        Cross-Entropy를 사용해 loss 계산하는 공식은 -1 * np.sum(Y * np.log(y_predict))임
        Y에서 답이 아닌 인덱스의 값은 모두 0이므로 곱셈을 하면 결국 0임.
        그래서 실질적으로 np.sum을 해서 나오는 값은 이미지의 실제 정답이 3이라 하면 np.log(y_precict(3))일 것임
        """

        epsilon = 1e-7
        y_predict += epsilon
        # y_predict 내 어떤 값이 너무너무 작은 값일 경우에(0에 수렴하는 값) 컴퓨터는 그냥 0으로 인식을 하기 때문에 log 연산을 하면
        # 'RuntimeWarning: divide by zero encountered in log' 라는 경고가 생김. 이를 방지하기 위해 아주 작은 수를 더해주었음

        if Y.ndim == 2: # 두 장 이상의 이미지를 매개변수로 받았을 때 Y와 y_predict의 shape는 (개수, 10)임
            loss_arr = -1 * np.sum(Y * np.log(y_predict), axis=1)
            loss = np.average(loss_arr) # 이미지가 여러 개일 경우, 이미지마다 나온 loss의 평균을 내야 함
        else: # 단 한 장의 이미지만 매개변수로 받았을 때 Y와 y_predict의 shape는 (10, )임
            loss = -1 * np.sum(Y * np.log(y_predict))

        return loss

    def Forward(self, X, Y):  # ScoreFunction과 Softmax, LossFunction를 적절히 활용해 y_predict 와 loss를 리턴시키는 함수. -> 직접 작성
        # 3.4
        """
        위에서 작성한 함수들을 사용하여 y_predict와 loss를 리턴함
        이 Forward 함수는 Optimization 함수에서 불러지는데,
        이 때 main.py에서 SN.Optimization(x_train, y_train, x_test, y_test)가 불러짐
        그러므로 이 Forward 함수의 매개변수 X, Y는 Optimization 함수에서 받았던 매개변수(이미지, 정답레이블)임
        """
        tmp = self.ScoreFunction(X)
        y_predict = self.Softmax(tmp)
        loss = self.LossFunction(y_predict, Y)

        return y_predict, loss

    def delta_Loss_Scorefunction(self, y_predict, Y):  # 제공.dL/dScoreFunction
        delta_Score = y_predict - Y
        return delta_Score

    def delta_Score_weight(self, delta_Score, X):  # 제공. dScoreFunction / dw .
        delta_W = np.dot(X.T, delta_Score) / X[0].shape
        return delta_W

    def delta_Score_bias(self, delta_Score, X):  # 제공. dScoreFunction / db .
        delta_B = np.sum(delta_Score) / X[0].shape
        return delta_B

    # delta 함수를 적절히 써서 delta_w, delta_b 를 return 하십시오.
    def BackPropagation(self, X, y_predict, Y):
        # 3.5
        """
        analytic gradient 방법에 입각해 dL/dw, dL/db를 계산하는 함수
        제공된 delta_함수들을 이용하여 계산함
        이 리턴값들은 이후에 loss값이 작아지도록 Weight값과 Bias값을 수정할 때 필요함
        """
        delta_score = self.delta_Loss_Scorefunction(y_predict, Y) # [이미지 개수, 10]의 형태
        if X.ndim == 1:  # 한개의 이미지를 대상으로 계산할 경우, 배열이 [784,]의 형태이므로 [1, 784]로 바꾸어 계산할 수 있도록 함
            X = X.reshape(1, 784)
        delta_W = self.delta_Score_weight(delta_score.reshape([X.shape[0], 10]), X)
        # 한 개의 이미지를 대상으로 계산할 땐 delta_score의 shape가 [10, ]임.
        # 그래서 한 개의 이미지일 때도 dot 연산이 가능하도록 매개변수로 delta_score.reshape([X.shape[0], 10])를 전달했음. X.shape[0]는 이미지 개수임
        delta_B = self.delta_Score_bias(delta_score, X)

        return delta_W, delta_B

    # 정확도를 체크하는 Accuracy 제공
    def Accuracy(self, X, Y):
        y_score = self.ScoreFunction(X)
        y_score_argmax = np.argmax(y_score, axis=1)
        if Y.ndim != 1: Y = np.argmax(Y, axis=1)
        accuracy = 100 * np.sum(y_score_argmax == Y) / X.shape[0]
        return accuracy

    # Forward와 BackPropagationAndTraining, Accuracy를 사용하여서 Training을 epoch만큼 시키고, 10번째 트레이닝마다
    # Training Set의 Accuracy 값과 Test Set의 Accuracy를 print 하십시오

    def Optimization(self, X_train, Y_train, X_test, Y_test, learning_rate=0.01, epoch=100):
        """
        :param X_train: 훈련 이미지
        :param Y_train: 훈련 이미지의 정답 레이블
        :param X_test: 시험 이미지
        :param Y_test: 시험 이미지의 정답 레이블
        :param learning_rate: loss의 값을 줄이기 위해 최적의 W와 b의 값을 조정할 때 얼마만큼 변화량을 주어 조정할 건지에 대한 수치임
        :param epoch: 훈련을 epoch번 시킴. 전체 데이터셋을 한번 돈 것을 1epoch이라 함
        이 함수에서는 그동안 작성했던 함수들을 이용하여 이미지 트레이닝을 시키고,
        10번의 epoch마다 loss와 Training_set의 Accuracy, Test_set의 Accuracy를 출력
        """
        for i in range(epoch):
            # 3.6
            y_predict, loss = self.Forward(X_train, Y_train)
            del_w, del_b = self.BackPropagation(X_train, y_predict, Y_train)
            change_w = self.W + (-1 * learning_rate * del_w)
            change_b = self.B + (-1 * learning_rate * del_b)
            self.SetParams(change_w, change_b) # weight와 bias를 새롭게 조정함

            # 함수 작성
            if i % 10 == 0:
                # 3.6 Accuracy 함수 사용
                # 훈련 이미지 & 시험 이미지에 대해 얼마나 답을 잘 맞추는지 정확도 각각 계산
                train_accuracy = self.Accuracy(X_train, Y_train)
                test_accuracy = self.Accuracy(X_test, Y_test)
                print(i, "번째 트레이닝")
                print('현재 Loss(Cost)의 값 : ', loss)
                print("Train Set의 Accuracy의 값 : ", train_accuracy)
                print("Test Set의 Accuracy의 값 :", test_accuracy)
