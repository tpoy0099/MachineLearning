import tensorflow as tf
import numpy as np
import sklearn.metrics as metrics
import random

#旋转变换矩阵
def rotation_mat(n_degree):
    theta = n_degree / 180 * np.pi
    m = np.mat(
            [[np.cos(theta), -np.sin(theta)],
             [np.sin(theta), np.cos(theta)]]
            )
    return m

#以两个中心点为基础生成二类样本
centra0 = np.mat([0, 0])
centra1 = np.mat([50, 50])

num_of_samples = 720
points_data = np.zeros((num_of_samples, 2))
labels_data = np.zeros((num_of_samples, 1))
for i in range(num_of_samples):
    rd = random.randint(0, 360)
    rmat = rotation_mat(rd)
    rxy = np.mat([random.randint(0, 35), random.randint(0, 35)])
    if i % 2 == 0:
        centra_p = centra0
        label_p = 0
    else:
        centra_p = centra1
        label_p = 1
    point = rxy * rmat + centra_p
    points_data[i] = point[0]
    labels_data[i] = label_p
    
#========================================================
# softmax with tensorflow
#========================================================

#为了防止出现exp值域溢出,基于标准差正规化数据
x_mul = 1
y_mul = 1
train_data = points_data.copy()  
if 1:  
    x_mul = points_data[:,0].std()
    y_mul = points_data[:,1].std()
    train_data[:,0] /= x_mul
    train_data[:,1] /= y_mul 
    
#添加权重系数1,即将Wx+B中的B合并进入W
ext_bias_w = np.ones((train_data.shape[0], 1))
train_data = np.column_stack((train_data, ext_bias_w))

#tensorflow graphs
X = tf.placeholder(tf.float32, [None, 3])
Y = tf.placeholder(tf.float32, [None, 1])
W = tf.Variable(tf.zeros([3, 2]))

#输出 n * 2 的矩阵, 行为每一个样本, 列为属于每一类(此处为两类)的数值
linear_part = tf.matmul(X, W)
e_trans_mat = tf.exp(linear_part)
#对于概率数矩阵, 按行求和, 即对于每个样本统计总概率数字, 输出 n * 1 的张量
sum_of_prob = tf.reshape(tf.reduce_sum(e_trans_mat, axis=1), (-1,1))
#将概率归一化, 输出 n * 2
P = e_trans_mat / sum_of_prob

#示性函数 1{Y=n} 当 Y == n 时输出 1, 否则输出 0
label_ts = tf.constant(list(range(2)), tf.float32)  
#输出 n * 2 的 {0,1} 张量 
l_mask = tf.cast(tf.equal(label_ts, Y), tf.float32)

#对数最大似然
log_loss = tf.reduce_mean(l_mask * tf.log(P))
theta_punish = (1/2) * tf.reduce_sum(W*W)
#加入衰减项目之后导致Bias不收敛, (体现为梯度消失)
#可能是由于"(1/2)"系数过大?
#剔除衰减项后参数估计正常
#loss = -log_loss + theta_punish
loss = -log_loss


learning_rate = 0.01
optim = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

for s in range(10000):
    sess.run(optim, {X:train_data, Y:labels_data})
    if s % 1000 == 0:
        print(sess.run(loss, {X:train_data, Y:labels_data}))

print(sess.run(W))

esti_W = sess.run(W)
x_coef = -esti_W[0][1] / esti_W[1][1]
bias_b = -esti_W[2][1] / esti_W[1][1]
print("y = %.2fx + %.2f" % (x_coef*y_mul/x_mul, bias_b*y_mul))


V_data = tf.placeholder(tf.float32, [None, 3])
ProbV = tf.exp(tf.matmul(V_data, W))
class_predict = tf.argmax(ProbV, axis=1)

predict_y = sess.run(class_predict, {V_data:train_data})

accu_rate = metrics.accuracy_score(labels_data, predict_y)
print(r"train data accuracy: %.2f%%" % (accu_rate*100))






