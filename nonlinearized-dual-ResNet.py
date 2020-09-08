# -*- coding: utf-8 -*-
# @Author: xuli shen
# @Date:   2020-06-10 15:54:06
# @Last Modified by:   bison
# @Last Modified time: 2020-09-08 14:44:26



import tensorflow as tf
from ResNets.utils import *
from ResNets.ResNet_minimax import network
from data_process import *
from model import * 
import sys
# from Email_des import email_
import glob
import numpy as np
import pandas as pd
from PIL import Image
import random
import time



from cvxopt import matrix
from cvxopt.solvers import qp
from cvxopt import solvers
solvers.options['show_progress'] = False

date = time.strftime("%Y%m%d%H%M%S", time.localtime())

num_of_group = int(sys.argv[2])
num_correct = 2 # for correct group

training_epochs = 2000
train_subgroup_batch = int(sys.argv[3])
test_batch_num = 100
display_step = 1

m = 2 * size  # label * size

save = True

# =====================================       DATA LOADER    ================================================
h,w = (100,80)
size = h * w 
# Receding_Hairline Wearing_Necktie Rosy_Cheeks Eyeglasses  Goatee  Chubby
#  Sideburns   Blurry  Wearing_Hat Double_Chin Pale_Skin   Gray_Hair   Mustache    Bald
label_class_all = ["Receding_Hairline","Wearing_Necktie","Rosy_Cheeks","Eyeglasses","Goatee","Chubby"
                ,"Sideburns","Blurry","Wearing_Hat","Double_Chin","Pale_Skin","Gray_Hair","Mustache","Bald"]

label_cls = label_class_all[int(sys.argv[1])]


pngs = sorted(glob.glob('./data/img_align_celeba/*.jpg'))
data = pd.read_table('./data/list_attr_celeba.txt',delim_whitespace=True,error_bad_lines=False)

eyeglasses = np.array(data[label_cls])
eyeglasses_cls = (eyeglasses + 1)/2 



label_glasses = np.zeros((202599,2))
correct_list = []
correct_list_test = []
false_list = []
false_list_test = []
for i in range(len(label_glasses)):
    if eyeglasses_cls[i] ==1 :
        label_glasses[i][1] = 1
        if i < 160000:
            correct_list.append(i)
        else:
            correct_list_test.append(i)
    else:
        label_glasses[i][0] = 1
        if i < 160000:
            false_list.append(i)
        else:
            false_list_test.append(i)

print(len(correct_list_test),len(false_list_test))

training_set_label = label_glasses[0:160000,:]
test_set_label = label_glasses[160000:,:]
training_set_cls = eyeglasses_cls[0:160000]
test_set_cls = eyeglasses_cls[160000:]

def create_trainbatch(num=10, channel = 0):
    train_num = random.sample(false_list,num)
    if channel == 0:
        train_set = np.zeros((num,h,w))
    else:
        train_set = np.zeros((num,h,w,3))

    train_set_label_ = []
    train_set_cls_ = []
    
    for i in range(num):
        img = Image.open(pngs[train_num[i]])   
        img_grey = img.resize((w,h))
        if channel == 0:
            img_grey = np.array(img_grey.convert('L'))
            train_set[i,:,:] = img_grey
        else:
            img_grey = np.array(img_grey)
            train_set[i,:,:,:] = img_grey
        
        train_set_label_.append(training_set_label[train_num[i]])
        train_set_cls_.append(training_set_cls[train_num[i]])

    train_set_label_new = np.array(train_set_label_)
    train_set_cls_new = np.array(train_set_cls_)
    
    return train_set/255, train_set_label_new, train_set_cls_new
    


def create_trainbatch_all_correct(num=10, channel = 0):

    train_num = random.sample(correct_list,num)
    if channel == 0:
        train_set = np.zeros((num,h,w))
    else:
        train_set = np.zeros((num,h,w,3))
    train_set_label_ = []
    train_set_cls_ = []
    n = 0
    for i in range(num):
        img = Image.open(pngs[train_num[i]])   
        img_grey = img.resize((w,h))
        if channel == 0:
            img_grey = np.array(img_grey.convert('L'))
            train_set[i,:,:] = img_grey
        else:
            img_grey = np.array(img_grey)
            train_set[i,:,:,:] = img_grey
        
        train_set_label_.append(training_set_label[train_num[i]])
        train_set_cls_.append(training_set_cls[train_num[i]])
    # if channel == 0:
    #     train_set = train_set.reshape(size,num).T   
    train_set_label_new = np.array(train_set_label_)
    train_set_cls_new = np.array(train_set_cls_)            
    
    return train_set/255, train_set_label_new, train_set_cls_new

def create_trainbatch_(num=10, channel = 0):
    train_num1 = random.sample(correct_list,int(num/2))
    train_num2 = random.sample(false_list,int(num/2))

    train_num = train_num1+train_num2
    if channel == 0:
        train_set = np.zeros((num,h,w))
    else:
        train_set = np.zeros((num,h,w,3))
    train_set_label_ = []
    train_set_cls_ = []
    n = 0
    for i in range(num):
        img = Image.open(pngs[train_num[i]])   
        img_grey = img.resize((w,h))
        if channel == 0:

            img_grey = np.array(img_grey.convert('L'))
            train_set[i,:,:] = img_grey
        else:
            img_grey = np.array(img_grey)
            train_set[i,:,:,:] = img_grey
        
        train_set_label_.append(training_set_label[train_num[i]])
        train_set_cls_.append(training_set_cls[train_num[i]])
    # if channel == 0:
    #     train_set = train_set.reshape(size,num).T   
    train_set_label_new = np.array(train_set_label_)
    train_set_cls_new = np.array(train_set_cls_)            
    
    return train_set/255, train_set_label_new, train_set_cls_new


def create_trainbatch_grad(num=200, channel = 0):
    train_num1 = random.sample(correct_list,int(10))
    train_num2 = random.sample(false_list,int(190))

    train_num = train_num1+train_num2

    if channel == 0:
        train_set = np.zeros((num,h,w))
    else:
        train_set = np.zeros((num,h,w,3))

    train_set_label_ = []
    train_set_cls_ = []
    n = 0
    for i in range(num):
        img = Image.open(pngs[train_num[i]])   
        img_grey = img.resize((w,h))
        if channel == 0:

            img_grey = np.array(img_grey.convert('L'))
            train_set[i,:,:] = img_grey
        else:
            img_grey = np.array(img_grey)
            train_set[i,:,:,:] = img_grey
        
        train_set_label_.append(training_set_label[train_num[i]])
        train_set_cls_.append(training_set_cls[train_num[i]])

    train_set_label_new = np.array(train_set_label_)
    train_set_cls_new = np.array(train_set_cls_)            
    
    return train_set/255, train_set_label_new, train_set_cls_new


def create_testset(num=100, channel = 0):
    
    test_num1 = random.sample(correct_list_test, num)
    test_num2 = random.sample(false_list_test, num)
    test_num= test_num1 + test_num2
    if channel == 0:
        test_set = np.zeros((num*2,h,w))
    else:
        test_set = np.zeros((num*2,h,w,3))
    
    test_set_label_ = []
    test_set_cls_ = []
    
    for i in range(num*2):
        img = Image.open(pngs[test_num[i]])   
        img_grey = img.resize((w,h))
        if channel == 0:
            img_grey = np.array(img_grey.convert('L'))
            test_set[i,:,:] = img_grey
        else:
            img_grey = np.array(img_grey)
            test_set[i,:,:,:] = img_grey
        
        test_set_label_.append(label_glasses[test_num[i]])
        test_set_cls_.append(eyeglasses_cls[test_num[i]])

    test_set_label_new = np.array(test_set_label_)
    test_set_cls_new = np.array(test_set_cls_)
    
    return test_set/255, test_set_label_new ,test_set_cls_new,test_set_cls_new.mean()*100



def create_testset_all( channel = 0):
    
    test_num1 = random.sample(correct_list_test, len(correct_list_test))
    test_num2 = random.sample(false_list_test, len(false_list_test))
    test_num= test_num1 + test_num2
    num = len(test_num)
    if channel == 0:
        test_set = np.zeros((num,h,w))
    else:
        test_set = np.zeros((num,h,w,3))
    
    test_set_label_ = []
    test_set_cls_ = []
    
    for i in range(num):
        img = Image.open(pngs[test_num[i]])   
        img_grey = img.resize((w,h))
        if channel == 0:
            img_grey = np.array(img_grey.convert('L'))
            test_set[i,:,:] = img_grey
        else:
            img_grey = np.array(img_grey)
            test_set[i,:,:,:] = img_grey
        
        test_set_label_.append(label_glasses[test_num[i]])
        test_set_cls_.append(eyeglasses_cls[test_num[i]])
    test_set_label_new = np.array(test_set_label_)
    test_set_cls_new = np.array(test_set_cls_)
    
    return test_set/255, test_set_label_new ,test_set_cls_new,test_set_cls_new.mean()*100


def create_testset_unbalanced( channel = 0):
    
    test_num1 = random.sample(correct_list_test, 10)
    test_num2 = random.sample(false_list_test, 190)
    test_num= test_num1 + test_num2
    num = len(test_num)
    if channel == 0:
        test_set = np.zeros((num,h,w))
    else:
        test_set = np.zeros((num,h,w,3))
    
    test_set_label_ = []
    test_set_cls_ = []
    
    for i in range(num):
        img = Image.open(pngs[test_num[i]])   
        img_grey = img.resize((w,h))
        if channel == 0:
            img_grey = np.array(img_grey.convert('L'))
            test_set[i,:,:] = img_grey
        else:
            img_grey = np.array(img_grey)
            test_set[i,:,:,:] = img_grey
            
        test_set_label_.append(label_glasses[test_num[i]])
        test_set_cls_.append(eyeglasses_cls[test_num[i]])
    test_set_label_new = np.array(test_set_label_)
    test_set_cls_new = np.array(test_set_cls_)
    
    return test_set/255, test_set_label_new ,test_set_cls_new,test_set_cls_new.mean()*100

# =====================================       DATA LOADER    ================================================



x = tf.placeholder(tf.float32, [None, h,w,3],name='x')
y = tf.placeholder(tf.float32, [None, 2],name='y') 

x_image = x

layer_fc2 = network(x=x_image)
layer_fc2 =tf.layers.flatten(layer_fc2)

weight_regularizer = tf.contrib.layers.l2_regularizer(0.0001)

W = tf.get_variable("w", shape=[5120, 2], initializer=tf.contrib.layers.variance_scaling_initializer())
b = tf.Variable(tf.constant(0.01, shape=[2]))

pred = tf.nn.softmax(tf.matmul(layer_fc2, W) + b) # Softmax
tf.add_to_collection('pred',pred)
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

# Minimize error using cross entropy

def cost_n_to_list(n,pred):
    list_ = []
    for _ in range(n):
        list_.append(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=pred)))
    return list_



def max_tensor(cost_combine):
    for i in range(len(cost_combine)-1):
        if i == 0:
            cost = cost_combine[i]
        else:
            cost = tf.maximum(cost, cost_combine[i+1])
    return cost


def grad_combine(cost_combine, var):
    gradss = []
    for i in range(len(cost_combine)):
        gradss.append(tf.gradients(xs=[var], ys=cost_combine[i]))
    return gradss

cost_combine = cost_n_to_list(num_of_group,pred)
cost_final = max_tensor(cost_combine)
grad_combine_ = grad_combine(cost_combine,var = W)
grad_combine_b = grad_combine(cost_combine,var = b)


def get_Gh(grad_list,cost_list,m):
    N = len(cost_list)
    G = np.zeros([N,m])
    b = []
    
    for i in range(N):
#         print(grad_list[i][0])
        g = grad_list[i][0].flatten()
        # print(g)
        G[i][:] = g
#         G[i][-1] = -1.0
        b.append(float(cost_list[i])) # add cost

    b = np.array(b)
#     print(b)
    GG = matrix(G)
    hh = matrix(b)
    
    return GG,hh


def cal_grad(grad_list, cost_list,m,size_in,size_out):
    
    N = len(cost_list)

    GG,hh = get_Gh(grad_list,cost_list,m)
    P = matrix(GG)*matrix(GG).T        
    q = -matrix(hh)
    
    G = matrix(-np.eye(N))
    h = matrix(np.zeros(N))
    A = matrix(np.ones([1,N]))
    b = matrix(np.ones([1]))
    
#     print(0)
    res = qp(P,q,G=G,h=h,A=A,b=b)
#     print(1)
    d = -np.array(GG).T.dot(np.array(res['x']))[:,0].reshape(size_in,size_out)
    # print('\n\n\n ++++++++++++++++++++++ \n',d)
    # print(len(d))
    return d

cost21 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=pred))


def grad_cost_list(sess,cost_combine,grad_combine_,grad_combine_b, num_correct=num_correct):
    grad_list = []
    grad_list_b = []

    cost_list = []

    n = len(cost_combine)
    train_x = []
    train_y = []
#     print(n)
    for i in range(n):
        if i < n - num_correct:
            batch_xs, batch_ys ,_ = create_trainbatch(train_subgroup_batch,3)
            c, g_W, g_b= sess.run([cost_combine[i], grad_combine_[i], grad_combine_b[i]], feed_dict={x: batch_xs, y: batch_ys})
#             print(c, g_W)
            grad_list.append(g_W)
            grad_list_b.append(g_b)

            cost_list.append(c)
            train_x.append(batch_xs)
            train_y.append(batch_ys)

        else:
#             print('hello')
            batch_xs, batch_ys ,_ = create_trainbatch_all_correct(train_subgroup_batch,3)
            c, g_W, g_b= sess.run([cost_combine[i], grad_combine_[i], grad_combine_b[i]], feed_dict={x: batch_xs, y: batch_ys})
#             print(c, g_W)
            grad_list.append(g_W)
            grad_list_b.append(g_b)

            cost_list.append(c)
            train_x.append(batch_xs)
            train_y.append(batch_ys)

    return np.array(train_x).reshape(n*train_subgroup_batch,h,w,3), np.array(train_y).reshape(n*train_subgroup_batch,2), grad_list, grad_list_b, cost_list


correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
accuracy_count = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(tf.clip_by_value(pred,1e-8,1.0)), reduction_indices=1))

def print_accuracy(accuracy, accuracy_count, cost):
    num = 200
    batch_xx,batch_yy,_,d= create_testset(num,3)

    batch_xx_correct = batch_xx[:num]
    batch_yy_correct = batch_yy[:num]
    batch_xx_false = batch_xx[num:]
    batch_yy_false = batch_yy[num:]

    feed_dict_test_T = {x: batch_xx_correct,
                      y: batch_yy_correct}

    feed_dict_test_F = {x: batch_xx_false,
                      y: batch_yy_false}    
    
    feed_dict_test = {x: batch_xx,
                      y: batch_yy}

    TP = accuracy_count.eval(feed_dict = feed_dict_test_T)
    FN = len(batch_xx_correct) - TP
    TN = accuracy_count.eval(feed_dict = feed_dict_test_F)
    FP = len(batch_xx_false) - TN

    recall = TP/(TP + FN + 1e-8)
    precision = TP/(TP + FP + 1e-8)
    F1 = 2 * ((precision*recall)/(precision + recall + 1e-8 ))
    print(TP,FN,TN,FP)
    print(recall, precision,F1, '\n')
    
    cost_f = cost.eval(feed_dict = feed_dict_test)

    batch_x, batch_y ,_ = create_trainbatch_(200,3)
    feed_dict = {x: batch_x,
                      y: batch_y}
    accuracy_train = accuracy.eval(feed_dict = feed_dict)

    
    
    return cost_f, F1 ,accuracy_train,d




def exchange(grads_and_vars,grad_,t):
    for i, (g, v) in enumerate([grads_and_vars[-1]]):
        if g is not None:
            grads_and_vars[-1] = (-grad_*t*0.0001, v)  # exchange gradients

def print_grad(grads_and_vars):

    for i, (g, v) in enumerate(grads_and_vars[-5:]):
        if g is not None:
            print(g)
            print(sess.run(np.max(g),feed_dict = {x: batch_xs21, y: batch_ys21}))



init = tf.global_variables_initializer()
acc_list = [0]
costs_list = []
acc_train_list = []
optimizer = tf.train.GradientDescentOptimizer(0.0001)

grads_and_vars_all = optimizer.compute_gradients(cost21)[:-2]

training_op_all = optimizer.apply_gradients(grads_and_vars_all)


grads_check = tf.gradients(cost, tf.trainable_variables()) 

show_all_variables()


def scale_t(max_grad):
    n = 0
    num = 0.1
    if max_grad > num:
        while max_grad > num:
            max_grad = max_grad * 0.1
            n -= 1


    return n

cost_before = 1000
cost_now = 10000
# Start training
with tf.Session() as sess:
    
    sess.run(init)
    t = 1

    print('\n ++  {}  ++ \n'.format(label_cls))
    m_saver = tf.train.Saver()
    for epoch in range(training_epochs):

        for i in range(1):


            
            batch_xs21, batch_ys21, grad_list, grad_list_b, cost_list = grad_cost_list(sess,cost_combine,grad_combine_,grad_combine_b)
            grad_ = cal_grad(grad_list, cost_list,m=5120*2,size_in=5120,size_out=2)
            grad_b = cal_grad(grad_list_b, cost_list,m=2,size_in=1,size_out=2)

            grad_ = grad_.astype(np.float32)
            grad_b = grad_b.astype(np.float32).reshape(2)

            max_grad_ = np.max(grad_)
            scale_grad_ = scale_t(max_grad_)


            max_grad_b = np.max(grad_b)
            scale_grad_b = scale_t(max_grad_b)

            grad_ = grad_ * (10 ** 5 )
            grad_b = grad_b * (10 ** 3 )


            print(scale_grad_,scale_grad_b)
            print(np.max(grad_),np.max(grad_b))
            print(np.min(abs(grad_)),np.min(abs(grad_b)))
            print('iter_{}  curr(F1_list) : '.format(epoch) , acc_list[-1])
            print('iter_{}  mean(F1_list) : '.format(epoch) , np.mean(acc_list[-10:-1]))
            print('iter_{}  max (F1_list) : '.format(epoch) , max(acc_list))

            training_op = optimizer.apply_gradients([((-grad_*t*1).astype(np.float32), W)])
            training_op_b = optimizer.apply_gradients([((-grad_b*t*1).astype(np.float32), b)])

            sess.run(training_op,feed_dict={x: batch_xs21, y: batch_ys21})
            sess.run(training_op_b,feed_dict={x: batch_xs21, y: batch_ys21})


            sess.run(training_op_all,feed_dict={x: batch_xs21, y: batch_ys21})
        
        if (epoch) % 1 == 0:
            cost_before = cost_now

            costs, acc, acc_train, _ = print_accuracy(accuracy, accuracy_count, cost)
            acc_list.append(acc)
            costs_list.append(costs)
            cost_now = costs

            acc_train_list.append(acc_train)
            print ("Epoch:", '%04d' % (epoch+1), "cost =", "{:.9f}".format(costs), 
                "stepsize =",t, "acc_test_F1 = {:.4f}".format(acc),"acc_train =", acc_train)
            
        if save:
            np.savez('./result/{}_F1__resnet_nonlinear_{}_{}.npz'.format(label_cls,num_of_group,date),test = acc_list)
            # if (epoch) % 5 == 0:
            #     m_saver.save(sess, "models/Res_MER", global_step=epoch)

    print ("Optimization Finished!")



print(acc_list)
print(costs_list)