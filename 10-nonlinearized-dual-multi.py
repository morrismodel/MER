# -*- coding: utf-8 -*-
# @Author: xuli shen
# @Date:   2020-06-10 15:54:06
# @Last Modified by:   bison
# @Last Modified time: 2020-09-08 14:35:50



import tensorflow as tf
import glob
import numpy as np
import pandas as pd
from PIL import Image
import random
import time
from data_process import *
from model import *

from cvxopt import matrix
from cvxopt.solvers import qp
from cvxopt import solvers
solvers.options['show_progress'] = False



num_of_group = 20
num_correct = 1  

training_epochs = 100
train_subgroup_batch = 10
test_batch_num = 100
display_step = 1

m = 2 * size 


def print_accuracy():
    
    a,b,c,d= create_testset()
    
    feed_dict_test = {x: a,
                      y: b}

    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    outcome = accuracy.eval(feed_dict = feed_dict_test)
#     print("Accuracy: {0:.5%}".format(outcome))
    cost_f = cost.eval(feed_dict = feed_dict_test)
    
    
    return cost_f, outcome,d


# tf Graph Input
x = tf.placeholder(tf.float32, [None, size]) 
y = tf.placeholder(tf.float32, [None, 2]) 

x_image = tf.reshape(x, [-1, w, h, 1])

filter_size1 = 5
num_filters1 = 16

filter_size2 = 5
num_filters2 = 36


layer_fc1 = new_fc_layer(input=x,
                         num_inputs=size,
                         num_outputs=6000,
                         use_relu=True)

layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=6000,
                         num_outputs=3000,
                         use_relu=True)


W = tf.Variable(tf.random_normal([3000, 2], stddev=0.01))

pred = tf.nn.softmax(tf.matmul(layer_fc2, W)) # Softmax
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))


def cost_n_to_list(n,pred):
    list_ = []
    for _ in range(n):
        list_.append(tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1)))
    return list_


def max_tensor(cost_combine):
    for i in range(len(cost_combine)-1):
        if i == 0:
            cost = cost_combine[i]
        else:
            cost = tf.maximum(cost, cost_combine[i+1])
    return cost


def grad_combine(cost_combine):
    gradss = []
    for i in range(len(cost_combine)):
        gradss.append(tf.gradients(xs=[W], ys=cost_combine[i]))
    return gradss

cost_combine = cost_n_to_list(num_of_group,pred)
cost_final = max_tensor(cost_combine)
grad_combine_ = grad_combine(cost_combine)

def get_Gh(grad_list,cost_list):
    N = len(cost_list)
    G = np.zeros([N,m])
    b = []
    
    for i in range(N):
#         print(grad_list[i][0])
        g = grad_list[i][0].flatten()
        G[i][:] = g
        b.append(float(cost_list[i])) # add cost

    b = np.array(b)

    GG = matrix(G)
    hh = matrix(b)
    
    return GG,hh


def cal_grad(grad_list, cost_list):
    
    N = len(cost_list)

    GG,hh = get_Gh(grad_list,cost_list)
    P = matrix(GG)*matrix(GG).T        
    q = -matrix(hh)
    
    G = matrix(-np.eye(N))
    h = matrix(np.zeros(N))
    A = matrix(np.ones([1,N]))
    b = matrix(np.ones([1]))
    
    res = qp(P,q,G=G,h=h,A=A,b=b)

    d = -np.array(GG).T.dot(np.array(res['x']))[:,0].reshape(-1,1).reshape(size,2)

    return d

cost21 = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
optimizer = tf.train.GradientDescentOptimizer(0.01)

def grad_cost_list(cost_combine,grad_combine_,num_correct=num_correct):
    grad_list = []
    cost_list = []
    n = len(cost_combine)

    for i in range(n):
        if i < n - num_correct:
            batch_xs, batch_ys ,_ = create_trainbatch(train_subgroup_batch)
            c, g_W = sess.run([cost_combine[i], grad_combine_[i]], feed_dict={x: batch_xs, y: batch_ys})

            grad_list.append(g_W)
            cost_list.append(c)
        else:

            batch_xs, batch_ys ,_ = create_trainbatch_all_correct(train_subgroup_batch)
            c, g_W = sess.run([cost_combine[i], grad_combine_[i]], feed_dict={x: batch_xs, y: batch_ys})
            grad_list.append(g_W)
            cost_list.append(c)
    return grad_list, cost_list



def exchange(grads_and_vars,grad_,t):
    for i, (g, v) in enumerate(grads_and_vars):
        if g is not None:
            grads_and_vars[i] = (-grad_*t*100, v)  # exchange gradients


init = tf.global_variables_initializer()
acc_list = []
costs_list = []

# Start training
with tf.Session() as sess:
    
    sess.run(init)
    t = 1

    for epoch in range(training_epochs):
        if (epoch+1) % 25 == 0:
            t = t * 0.1
        for i in range(5):
            
            grad_list, cost_list = grad_cost_list(cost_combine,grad_combine_)
            grad_ = cal_grad(grad_list, cost_list)
    
            a,b,c,d= create_testset(test_batch_num)
            feed_dict_test = {x: a,
                      y: b}

            now = sess.run(cost_final, feed_dict=feed_dict_test)
            ww = sess.run(W, feed_dict=feed_dict_test)
            
            grad_ = tf.cast(grad_, tf.float32)
            grads_and_vars = optimizer.compute_gradients(cost21, W)
            exchange(grads_and_vars,grad_,t)
            training_op = optimizer.apply_gradients(grads_and_vars)
            batch_xs21, batch_ys21 ,_ = create_trainbatch(train_subgroup_batch)
            sess.run(training_op,feed_dict={x: batch_xs21, y: batch_ys21})
        
        if (epoch+1) % display_step == 0:
            
            costs, acc, _ = print_accuracy()
            acc_list.append(acc)
            costs_list.append(costs)
            

            print ("Epoch:", '%04d' % (epoch+1), "cost =", "{:.9f}".format(costs), "stepsize =",t, "acc =", acc)

    print ("Optimization Finished!")


print(acc_list)
print(costs_list)

