import random
import tensorflow as TF
import numpy as NP
import matplotlib.pyplot as PLT


x_data = [[random.randint(0, 100) / 1.1] for _ in range(100)]
y_data = [[i[0] * 20.5 + 12.8 + random.randint(0, 200)] for i in x_data]

#输入变量
X = TF.placeholder(TF.float32, [None, 1])
Y = TF.placeholder(TF.float32, [None, 1])
#待估计变量
Cx = TF.Variable(TF.zeros([1,1]))
B = TF.Variable(TF.zeros([1,1]))

#模型多项式
linear_model = TF.matmul(X, Cx) + B
#损失函数
mse_loss = TF.reduce_mean(TF.square(Y - linear_model))

#学习率
learning_rate = 0.1
#使用梯度下降优化
linear_train_op = TF.train.GradientDescentOptimizer(
        learning_rate = learning_rate
        )
#正则化梯度下降率
linear_train_op = TF.contrib.estimator.clip_gradients_by_norm(
        linear_train_op, 
        5.0
        )
linear_train_op = linear_train_op.minimize(mse_loss)

#初始化tensorflow全局变量
sess = TF.Session()
sess.run(TF.global_variables_initializer())


train_times = list()
mse_data = list()
for s in range(100):
    sess.run(linear_train_op, {X:x_data, Y:y_data})
    if s % 2 == 0:
        tmp_e = sess.run(mse_loss, {X:x_data, Y:y_data})
        mse_data.append(tmp_e)
        train_times.append(s)

x_coef = sess.run(Cx)
x_const = sess.run(B)
predict_y = sess.run(linear_model, {X:x_data})
print("GD resault : Y = %.2f * X + %.2f" % (x_coef, x_const))

fig = PLT.figure()
ax1 = fig.add_subplot(2,1,1)
ax1.set_xlabel("train_times")
ax1.set_ylabel("mse")
ax1.plot(train_times, mse_data)

ax2 = fig.add_subplot(2,1,2)
ax2.set_title("data plot")
ax2.plot(x_data, y_data, "ro")
ax2.plot(x_data, predict_y.flatten(), "-")

PLT.show()
    
    
