import numpy as np
import pandas as pd
import tensorflow as tf

#----------------------- data -------------------------#
train_data = pd.read_excel('train.xlsx')
train_data = np.array(train_data,dtype = 'float32')
train_input = train_data[:,0:-1]
train_output = train_data[:,-1]
##train_output = np.reshape(train_output,[len(train_output),1])

height, width = train_data.shape
batch1 = np.zeros([height - 5, 5, 4])   #sample [5,4]
batch2 = np.zeros([height - 5, 1, 1])   #label  [1,1]
for i in range(height - 5) :
    batch1[i,:,:] = train_input[i:i+5,:]
    batch2[i,:,:] = train_output[i+5]
#------------------------------------------------------#

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32,[None,5,4])
y_ = tf.placeholder(tf.float32,[None,1,1])

#----------------------- first layer -------------------------#
W_conv1 = tf.Variable(tf.truncated_normal([2, 4, 40], stddev = 0.1))  #[]
##print(W_conv1.get_shape())
b_conv1 = tf.Variable(tf.constant([0.1], shape = [40]))
##print(b_conv1.get_shape())

##conv1 = tf.nn.conv1d(x, W_conv1, stride = 1, padding = 'VALID')
conv1 = tf.nn.conv1d(x, W_conv1, stride = 1, padding = 'VALID') + b_conv1
h_conv1 = tf.nn.relu(conv1)       #output shape    [None, 4, 20]
##h_pool = tf.nn.avg_pool(h_conv1, ksize = [1, 1, 2, 1], strides = [1, 1, 2, 1],
##                        padding = 'VALID')
#-------------------------------------------------------------#

#----------------------- second layer -------------------------#
W_conv2 = tf.Variable(tf.truncated_normal([2, 40, 80], stddev = 0.1))
b_conv2 = tf.Variable(tf.constant([0.1], shape = [80]))

conv2 = tf.nn.conv1d(h_conv1, W_conv2, stride = 1, padding = 'VALID')
h_conv2 = tf.nn.relu(conv2)      #output shape     [None, 3, 40]
#--------------------------------------------------------------#

W_fc1 = tf.Variable(tf.truncated_normal([3* 80, 200], stddev = 0.1))
b_fc1 = tf.Variable(tf.constant([0.1], shape = [200]))
h_conv2_flat = tf.reshape(h_conv2, [-1, 3* 80])
keep_prob = tf.placeholder(tf.float32)

h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = tf.Variable(tf.truncated_normal([200, 200], stddev = 0.1))
b_fc2 = tf.Variable(tf.constant([0.1], shape = [200]))

h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2)  + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

W_fc3 = tf.Variable(tf.truncated_normal([200, 1], stddev = 0.1))
b_fc3 = tf.Variable(tf.constant([0.1], shape = [1]))

y_pre = tf.matmul(h_fc2_drop, W_fc3)  + b_fc3

cross_entropy = tf.sqrt(tf.reduce_mean(tf.square(y_pre - y_)))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


tf.global_variables_initializer().run()

def train_rmse() :
    global y_pre
    prediction = sess.run(y_pre,feed_dict = {x : batch1, keep_prob : 1})

    train_rmse = tf.sqrt(tf.reduce_mean(tf.square(prediction - y_)))

    result = sess.run(train_rmse, feed_dict = {x : batch1,
                                               y_ : batch2,
                                               keep_prob : 1})
    return result


for i in range(20000) :
    sess.run(train_step, feed_dict={x : batch1,
                                y_: batch2,
                                keep_prob : 0.5})
##    prediction = sess.run(y_pre,)
##    print(i)
    if i % 50 == 0 :
        print(train_rmse())
