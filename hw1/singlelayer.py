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
        X는 [이미지 개수, 784]인 배열, W는 [784,10(label 개수)]인 배열이므로 np.dot() 함수를 이용하여 곱셈을 하고
        가중치인 B를 더해주면 결과로 나올 ScoreMatrix는 [이미지 개수, 10]인 배열임
        ScoreMatrix에서 최대값이 있는 index가 추론한 값이 될 것임
        """

        ScoreMatrix = np.dot(X, self.W) + self.B

        return ScoreMatrix

    def Softmax(self, ScoreMatrix):  # 제공.
        if ScoreMatrix.ndim == 2:
            temp = ScoreMatrix.T
            temp = temp - np.max(temp, axis=0)  # 가장 점수가 높은 인덱스의 값을 0으로 만들기 위해 np.max(temp,axis=0)을 뺌
            y_predict = np.exp(temp) / np.sum(np.exp(temp), axis=0)  # 값이 0이면 exp 연산했을 때 결과가 1임
            return y_predict.T
        temp = ScoreMatrix - np.max(ScoreMatrix, axis=0)
        expX = np.exp(temp)
        y_predict = expX / np.sum(expX)
        return y_predict

    def LossFunction(self, y_predict, Y):  # Loss Function을 구하십시오 -> 직접 작성
        # 3.3
        """ 얘 수정!!!!!!!!!!!!!!!!!!!!!!
        :param y_predict: Softmax 함수를 이용해서 얻은 결과값임.
        :param Y: 해당 사진 데이터의 정답 레이블임.
        :return: 예를 들어 사진의 정답값이 3인데 y_predict에서 가장 큰 값이 1이었을 경우, 이 정답을 얼마나 제대로 맞추지 못했는지를 알려주는 척도임
        즉, 리턴값인 loss가 작을 수록 잘 추측한 것임
        """

        if Y.ndim == 2: # 사진 여러 장일 경우, Y.shape는 [사진 개수, 10]임. 각 사진별로 정답값을 tmp에 저장함
            tmp = np.argmax(Y, axis=1)
            loss = np.zeros([tmp.size])
            for i in range(tmp.size):
                loss[i] = -1 * np.log10(y_predict[i][tmp[i]])  # 강의 때 배운 loss 계산식임
        else: # 사진 한 장일 경우 Y.shape는 [10, ]임
            tmp = np.argmax(Y)
            loss = -1 * np.log10(y_predict[0][tmp])

        return loss

    def Forward(self, X, Y):  # ScoreFunction과 Softmax, LossFunction를 적절히 활용해 y_predict 와 loss를 리턴시키는 함수. -> 직접 작성
        # 3.4
        """
        위에서 작성한 함수들을 사용하여 y_predict와 loss를 리턴함
        이 함수는 Optimization 함수에서 사용되는데,
        이 때 main.py에서 SN.Optimization(x_train, y_train, x_test, y_test)가 불러진다.
        그러므로 이 함수의 매개변수 X, Y는 Optimization 함수에서 받은 매개변수를 이어서 사용하는 것이다.
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
        이 리턴값들은 이후에 loss값이 작아지도록 Weight값을 수정할 때 필요함
        """
        if X.ndim == 1:  # 한개의 사진을 대상으로 계산할 경우, 배열이 [784,]의 형태이므로 [1, 784]로 바꾸어 계산할 수 있도록 함
            X = X.reshape(1, X.size)
        delta_score = self.delta_Loss_Scorefunction(y_predict, Y)
        delta_W = delta_score * self.delta_Score_weight(delta_score, X)
        delta_B = delta_score * self.delta_Score_bias(delta_score, X)

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
        for i in range(epoch):
            # 3.6
            y_predict, loss = self.Forward(X_train[i], Y_train[i])
            del_w, del_b = self.BackPropagation(X_train[i], y_predict, Y_train[i])
            self.SetParams(self.W + (-learning_rate * del_w), self.B + (-learning_rate * del_b))

            # 함수 작성
            if i % 10 == 0:
                # 3.6 Accuracy 함수 사용
                train_accuracy = self.Accuracy(X_train[i], Y_train[i])
                test_accuracy = self.Accuracy(X_test[i], Y_test[i])
                print(i, "번째 트레이닝")
                print('현재 Loss(Cost)의 값 : ', loss)
                print("Train Set의 Accuracy의 값 : ", train_accuracy)
                print("Test Set의 Accuracy의 값 :", test_accuracy)
