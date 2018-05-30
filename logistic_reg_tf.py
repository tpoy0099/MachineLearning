import numpy as NP
import tensorflow as TF
import random
import matplotlib.pyplot as PLT
from tensorflow.python import debug as tf_debug

#旋转变换矩阵
def rotation_mat(n_degree):
    theta = n_degree / 180 * NP.pi
    m = NP.mat(
            [[NP.cos(theta), -NP.sin(theta)],
             [NP.sin(theta), NP.cos(theta)]]
            )
    return m

#以两个中心点为基础生成二类样本
centra0 = NP.mat([0, 0])
centra1 = NP.mat([50, 50])

points_data = NP.zeros((720, 2))
labels_data = NP.zeros((720, 1))
for i in range(points_data.shape[0]):
    rd = random.randint(0, 360)
    rmat = rotation_mat(rd)
    rxy = NP.mat([random.randint(0, 35), random.randint(0, 35)])
    if i % 2 == 0:
        centra_p = centra0
        label_p = 0
    else:
        centra_p = centra1
        label_p = 1
    point = rxy * rmat + centra_p
    points_data[i] = point[0]
    labels_data[i] = label_p
    
#===========================================================
#===========================================================

#为了防止出现exp值域溢出,基于标准差正规化数据
x_mul = 1
y_mul = 1
train_data = points_data.copy()  
if 1:  
    x_mul = points_data[:,0].std()
    y_mul = points_data[:,1].std()
    train_data[:,0] /= x_mul
    train_data[:,1] /= y_mul 

#tensorflow
X = TF.placeholder(TF.float32, [None, 2], name="X")
Y = TF.placeholder(TF.float32, [None, 1], name="Y")
W = TF.Variable(TF.zeros([2, 1]), name="W")
B = TF.Variable(TF.zeros([1, 1]), name="B")

#多项式
linear_mod = TF.matmul(X, W) + B
#将多项式函数输出值用S函数(这里选用logit), 映射到值域(0, 1)
logic_mod = 1 / (1 + TF.exp(-linear_mod))

#最大似然对数损失
#loss = -TF.reduce_mean(Y * TF.log(logic_mod) + (1 - Y) * TF.log(1 - logic_mod))
#化简形式
loss_p0 = Y * -linear_mod
loss_p1 = TF.log(1 + TF.exp(-linear_mod))
loss = TF.reduce_mean(loss_p1 - loss_p0)

#学习率
learning_rate = 0.01
#使用梯度下降优化
gdop = TF.train.GradientDescentOptimizer(learning_rate)
#正则化梯度下降率
gdop = TF.contrib.estimator.clip_gradients_by_norm(gdop, 2.0)
gdop = gdop.minimize(loss)

sess = TF.Session()

sess.run(TF.global_variables_initializer())

#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
for s in range(10000):
    sess.run(gdop, {X:train_data, Y:labels_data})

print(">>>>> with tensorflow")    
esti_W = sess.run(W)
esti_B = sess.run(B)
x_coef = -esti_W[0][0] / esti_W[1]
bias_b = -esti_B[0][0] / esti_W[1]
print("y = %.2fx + %.2f" % (x_coef*y_mul/x_mul, bias_b*y_mul))

if 1:
    fig = PLT.figure()
    ax1 = fig.add_subplot(111)
    for i in range(points_data.shape[0]):
        x, y = points_data[i]
        if labels_data[i] == 1:
            ax1.plot(x, y, 'ro')
        else:
            ax1.plot(x, y, 'go')  
    #empirical estimate
    # x + y - 50 = 0
    x_range = [x[0] for x in points_data]
    esti_y = [(x_coef * x[0] / x_mul + bias_b)*y_mul for x in points_data]
    ax1.plot(x_range, esti_y, 'k-')

#=======================================================
#tensorflow senior APIs
#=======================================================
TF.logging.set_verbosity(TF.logging.ERROR)

def data_feeder(xy_data, label_data):
    data_dc = {"x":xy_data[:,0], "y":xy_data[:,1]}
    ts = TF.data.Dataset.from_tensor_slices((data_dc, label_data.reshape([1,-1])[0]))
    ts = ts.shuffle(label_data.shape[1]).repeat().batch(label_data.shape[1])
    
    t_data, t_label = ts.make_one_shot_iterator().get_next() 
    
    return t_data, t_label

clf_feature_cols = [
        TF.feature_column.numeric_column("x"),
        TF.feature_column.numeric_column("y")
        ]

clf_opt = TF.train.GradientDescentOptimizer(0.01)
clf_opt = TF.contrib.estimator.clip_gradients_by_norm(clf_opt, 2.0)

clf_model = TF.estimator.LinearClassifier(
        feature_columns=clf_feature_cols, 
        optimizer=clf_opt
        )
#tf的高级api对于过大的输入项(如使得exp(x)超出浮点精度的x)
#处理方式应该是直接进行了裁剪, 这会导致截距项bias发生变化
#所以依然需要对输入变量做正规化
clf_model.train(
        input_fn = lambda :data_feeder(train_data, labels_data), 
        steps=10000
        )

clf_x_coef = clf_model.get_variable_value('linear/linear_model/x/weights')[0][0]
clf_y_coef = clf_model.get_variable_value('linear/linear_model/y/weights')[0][0]
clf_b_bias = clf_model.get_variable_value('linear/linear_model/bias_weights')[0]

print(">>>>> with tensorflow senior APIs")  
print("y = %.2fx + %.2f" % 
      (-clf_x_coef/clf_y_coef*y_mul/x_mul, -clf_b_bias/clf_y_coef*y_mul)
      )

#=======================================================
#sklearn
#=======================================================
print(">>>> with sklearn")
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(points_data, labels_data)
skl_x_coef = lr.coef_[0][0] / -lr.coef_[0][1]
skl_b_bias = lr.intercept_[0] / -lr.coef_[0][1]
print("y = %.2fx + %.2f" % (skl_x_coef, skl_b_bias))


