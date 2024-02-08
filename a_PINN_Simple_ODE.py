import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import warnings
import os


os.environ["TF_NUM_INTEROP_THREADS"] = "4"  # TensorFlow와 다른 라이브러리 간의 스레드 수
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"   # TensorFlow 내부의 연산 스레드 수



# Define Networks and Parameters

NN = tf.keras.models.Sequential([
    tf.keras.layers.Input((1,)),
    tf.keras.layers.Dense(units=32, activation='tanh'),
    tf.keras.layers.Dense(units=20, activation='tanh'),
    tf.keras.layers.Dense(units=6, activation='tanh'),
    tf.keras.layers.Dense(units=1)
])

NN.summary()

# Define Optimizer
optm = tf.keras.optimizers.Adam(learning_rate = 0.001)

# Define ODE System

def ode_system(t,func_NN):
    t=t.reshape(-1,1) #열 벡터로 변환
    t=tf.constant(t, dtype=tf.float32) #t 열벡터에 있는 값을 tensorflow 상수 텐서로 변환
    t_0=tf.zeros((1,1)) #모든 값이 0인 1x1 tenflow 상수 텐서 생성
    one = tf.ones((1, 1)) #모든 값이 1인 1x1 tenflow 상수 텐서 생성

    with tf.GradientTape() as tape:
        tape.watch(t) #자동 미분을 적용하기 위한 작업

        u = func_NN(t) # ODE에 대한 수학적 대리모델 의미를 가지는 인공신경망
        u_t = tape.gradient(u, t) # # ODE에 대한 수학적 대리모델 의미를 가지는 인공신경망에 대한 기울기

    ode_loss = u_t - tf.math.cos(2 * np.pi * t) #ODE loss
    IC_loss = func_NN(t_0) - one # 초기조건 loss

    square_loss = tf.square(ode_loss) + tf.square(IC_loss)
    total_loss = tf.reduce_mean(square_loss)

    return total_loss


train_t = (np.random.rand(30) * 2).reshape(-1, 1) #30개의 난수 생성후에 2를 곱해서 0~2 스케일링 (0~2초) 후에 열벡터로 변환
train_loss_record = [] # loss 값 저장할 리스트 생성

for itr in range(6000): #6000번 반복
    with tf.GradientTape() as tape:
        train_loss = ode_system(train_t, NN) # 앞서 지정한 인공신경망에 생성한 난수 값을 입력하여 loss 값 계산
        train_loss_record.append(train_loss) # loss 값 저장

        grad_w = tape.gradient(train_loss, NN.trainable_variables) # 파라미터의 기울기 계산
        optm.apply_gradients(zip(grad_w, NN.trainable_variables)) #파라미터의 기울기 기반으로 파라미터 최적화

    if itr % 1000 == 0: #1000번 학습때 마다 loss값 출력
        print(train_loss.numpy())

# Exact Solution과 비교
plt.figure(figsize=(10, 8))
plt.plot(train_loss_record)
plt.show()

test_t = np.linspace(0, 2, 100)

train_u = np.sin(2*np.pi*train_t)/(2*np.pi) + 1
true_u = np.sin(2*np.pi*test_t)/(2*np.pi) + 1
pred_u = NN.predict(test_t).ravel()

plt.figure(figsize = (10,8))
plt.plot(train_t, train_u, 'ok', label = 'Train')
plt.plot(test_t, true_u, '-k',label = 'True')
plt.plot(test_t, pred_u, '--r', label = 'Prediction')
plt.legend(fontsize = 15)
plt.xlabel('t', fontsize = 15)
plt.ylabel('u', fontsize = 15)
plt.show()

