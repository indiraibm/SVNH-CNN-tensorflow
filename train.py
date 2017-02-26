import tensorflow as tf
import random
import numpy as np
from datetime import datetime

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.05)
	return tf.Variable(initial)
def bias_variable(shape):
	initial = tf.constant(0.05, shape = shape)
	return tf.Variable(initial)
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 4, 4, 1],strides=[1, 2, 2, 1], padding='SAME')
def change_format(case):
	output={'x':[],'y':[]}
	X = np.asarray(case['X'])
	Y = np.asarray(case['y'])
	numImages = X.shape[3]
	for i in xrange(numImages):
		output['x'].append(np.divide(X[:,:,:,i],128,dtype=float))
		tem = np.zeros(10)
		tem[Y[i]%10] = 1
		output['y'].append(tem)
	output['x'] = np.array(output['x'])
	output['y'] = np.array(output['y'])
	return output
def batch_gen(start,end,case,rand_num):
	caseList = rand_num[start:end]
	batch={'x':case['x'][caseList],'y':case['y'][caseList]}
	return batch

def random_gen(num):
	i = range(num)
	random.shuffle(i)
	return i

train = np.load("train.npy", encoding='bytes')
test = np.load("test.npy", encoding='bytes')
train = {'x':train.tolist()[b'x'],'y':train.tolist()[b'y']}
test = {'x':test.tolist()[b'x'],'y':test.tolist()[b'y']}
train_np = train
test_np = test
num_list_train = range(len(train['y']))

sess = tf.InteractiveSession()
print("creat sess")
x = tf.placeholder(tf.float32, shape = [None,32,32,3])
y_ = tf.placeholder(tf.float32, shape = [None,10])
keep_prob = tf.placeholder(tf.float32)

W_conv11 = weight_variable([5, 5, 3, 64])
b_conv11 = bias_variable([64])
h_conv11 = tf.nn.relu(conv2d(x, W_conv11) + b_conv11)
W_conv12 = weight_variable([5, 5, 64, 64])
b_conv12 = bias_variable([64])
h_conv12 = tf.nn.relu(conv2d(h_conv11, W_conv12) + b_conv12)

h_pool1 = max_pool_2x2(h_conv12)

W_conv21 = weight_variable([5, 5, 64, 64])
b_conv21 = bias_variable([64])
h_conv21 = tf.nn.relu(conv2d(h_pool1, W_conv21) + b_conv21)
W_conv22 = weight_variable([5, 5, 64, 32])
b_conv22 = bias_variable([32])
h_conv22 = tf.nn.relu(conv2d(h_conv21, W_conv22) + b_conv22)

h_pool2 = max_pool_2x2(h_conv22)

W_conv31 = weight_variable([5, 5, 32, 32])
b_conv31 = bias_variable([32])
h_conv31 = tf.nn.relu(conv2d(h_pool2, W_conv31) + b_conv31)
W_conv32 = weight_variable([5, 5, 32, 32])
b_conv32 = bias_variable([32])
h_conv32 = tf.nn.relu(conv2d(h_conv31, W_conv32) + b_conv32)

h_pool3 = max_pool_2x2(h_conv32)

h_pool3_flat = tf.reshape(h_pool3, [-1, 4*4*32])

W_fc1 = weight_variable([4*4*32, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())
print("global_var_init")
num_list_test_fix = np.random.choice(len(test['y']),100,replace=False)
#print(num_list_test_fix)
test_batch_fix = batch_gen(0,100,test_np,num_list_test_fix)
start = datetime.now() 

print("session start")
#for i in range(5000):
for i in range(len(train['y']) * 1)[0::50]:
	batch = batch_gen((i),min((i+50),len(train['y'])),train_np,num_list_train)
	num_list_test_ran = np.random.choice(len(test['y']),100,replace=False)
	test_batch_ran = batch_gen(0,100,test_np,num_list_test_ran)
	if i%100 == 0 or True:
		test_accuracy_fix = accuracy.eval(feed_dict={x:test_batch_fix['x'],y_:test_batch_fix['y'],keep_prob:1.0})
		test_accuracy_ran = accuracy.eval(feed_dict={x:test_batch_ran['x'],y_:test_batch_ran['y'],keep_prob:1.0})
		train_accuracy = accuracy.eval(feed_dict={x:batch['x'],y_:batch['y'],keep_prob:1.0})
		now = datetime.now()
		diff = now-start
		ds = diff.seconds
		h = int(ds / 60 / 60)
		m = int(ds / 60) % 60
		s = int(ds) % 60
	train_step.run(feed_dict={x:batch['x'],y_:batch['y'],keep_prob:0.5})
	print("step: {0:5d} train: {1:3d}% fix: {2:3d}% ran: {3:3d}% {4:2d}:{5:2d}:{6:2d}".format(i,int(train_accuracy*100),int(test_accuracy_fix*100),int(test_accuracy_ran*100),h,m,s))
print("finish")
train_accuracy = 0
start = datetime.now() 
count=0
for i in range(len(test['y']))[0::50]:
	if i+50>len(test['y']):
		break
	count = count + 1
	now = datetime.now()
	diff = now-start
	ds = diff.seconds
	h = int(ds / 60 / 60)
	m = int(ds / 60) % 60
	s = int(ds) % 60
	batch = batch_gen((i),min((i+50),len(test['y'])),test_np,range(len(test['y'])))
	train_accuracy = train_accuracy + accuracy.eval(feed_dict={x:batch['x'],y_:batch['y'],keep_prob:1.0})
	print("step: {0:5d} train: {1:6d}% {2:2d}:{3:2d}:{4:2d}".format(i,int(train_accuracy*100),h,m,s))
train_accuracy = train_accuracy/count
print ("final train_accuracy: ",train_accuracy)
saver = tf.train.Saver()
saver.save(sess,"c:\\Users\\user\\Desktop\\SVNH\\Save.ckpt",global_step = int(train_accuracy*100))
