# -*- coding: utf-8 -*-
"""
Created on Mon May 10 12:19:41 2021

@author: tians
"""

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import pylab
import tensorflow as tf

# visulaize
from scipy.interpolate import griddata
from numpy import linspace, meshgrid
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

#dns_100
#dataset 99-107


tf.reset_default_graph()
random.seed(9001)

def conv_layer(input, num_input_channels, conv_filter_size, num_filters, padding='SAME', relu=True):
    
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters]) #[3, 3, 1, 1]
    biases = create_biases(num_filters)
    layer = tf.nn.conv2d(input = input, filter = weights, strides=[1, 1, 1, 1], padding=padding)
    layer += biases

    if relu:
        layer = tf.nn.relu(layer)
        
    return layer

def create_weights(shape):
    
    # weights = np.zeros(shape)

    # return weights
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05, dtype=tf.dtypes.float64))

def create_biases(size):
    
    return tf.Variable(tf.constant(0.05, shape=[size], dtype=tf.dtypes.float64))


def calculate_div(u, v, w, xderi_weights, yderi_weights, zderi_weights):
    
        
    # ------------------------equation 1
    temp = u
    u_x = tf.nn.conv2d(tf.reshape(temp, [69, 132, 132, 1]), filter = xderi_weights, strides=[1, 1, 1, 1], padding='SAME')
    
    u_x = tf.reshape(u_x, [69, 132, 132])
        
    temp = u
    u_y = tf.nn.conv2d(tf.reshape(temp, [69, 132, 132, 1]), filter = yderi_weights, strides=[1, 1, 1, 1], padding='SAME')
    
    u_y = tf.reshape(u_y, [69, 132, 132])
    
    
    temp = tf.transpose(u, [1, 0, 2])     
    temp = tf.reshape(temp, [132, 69, 132, 1])     #add transpose
    u_z = tf.nn.conv2d(temp, filter = zderi_weights, strides=[1, 1, 1, 1], padding='SAME')
    u_z = tf.transpose(u_z, [1, 0, 2, 3]) 
    u_z = tf.reshape(u_z, [69, 132, 132])
    
    u_div = u* u_x + v* u_y + w* u_z
    
    
    # ------------------------equation 2
    temp = v
    v_x = tf.nn.conv2d(tf.reshape(temp, [69, 132, 132, 1]), filter = xderi_weights, strides=[1, 1, 1, 1], padding='SAME')
    
    v_x = tf.reshape(v_x, [69, 132, 132])
        
    temp = v
    v_y = tf.nn.conv2d(tf.reshape(temp, [69, 132, 132, 1]), filter = yderi_weights, strides=[1, 1, 1, 1], padding='SAME')
    
    v_y = tf.reshape(v_y, [69, 132, 132])
    
    temp = tf.transpose(v, [1, 0, 2])     
    temp = tf.reshape(temp, [132, 69, 132, 1])     #add transpose
    v_z = tf.nn.conv2d(temp, filter = zderi_weights, strides=[1, 1, 1, 1], padding='SAME')
    v_z = tf.transpose(v_z, [1, 0, 2, 3]) 
    v_z = tf.reshape(v_z, [69, 132, 132])
    
    v_div = u* v_x + v* v_y + w* v_z
    
    
    # ------------------------equation 3
    temp = w
    w_x = tf.nn.conv2d(tf.reshape(temp, [69, 132, 132, 1]), filter = xderi_weights, strides=[1, 1, 1, 1], padding='SAME')
    
    w_x = tf.reshape(w_x, [69, 132, 132])
        
    temp = w
    w_y = tf.nn.conv2d(tf.reshape(temp, [69, 132, 132, 1]), filter = yderi_weights, strides=[1, 1, 1, 1], padding='SAME')
    
    w_y = tf.reshape(w_y, [69, 132, 132])
    
    temp = tf.transpose(w, [1, 0, 2])     
    temp = tf.reshape(temp, [132, 69, 132, 1])     #add transpose
    w_z = tf.nn.conv2d(temp, filter = zderi_weights, strides=[1, 1, 1, 1], padding='SAME')
    w_z = tf.transpose(w_z, [1, 0, 2, 3]) 
    w_z = tf.reshape(w_z, [69, 132, 132])
    
    w_div = u* w_x + v* w_y + w* w_z
    
    return u_div, v_div, w_div


def calculate_laplacian(u, v, w, xxderi_weights, yyderi_weights, zzderi_weights):
    
        
    # ------------------------equation 1
    temp = u
    u_xx = tf.nn.conv2d(tf.reshape(temp, [69, 132, 132, 1]), filter = xxderi_weights, strides=[1, 1, 1, 1], padding='SAME')
    
    u_xx = tf.reshape(u_xx, [69, 132, 132])
        
    temp = u
    u_yy = tf.nn.conv2d(tf.reshape(temp, [69, 132, 132, 1]), filter = yyderi_weights, strides=[1, 1, 1, 1], padding='SAME')
    
    u_yy = tf.reshape(u_yy, [69, 132, 132])
    
    
    temp = tf.transpose(u, [1, 0, 2])     
    temp = tf.reshape(temp, [132, 69, 132, 1])     #add transpose
    u_zz = tf.nn.conv2d(temp, filter = zzderi_weights, strides=[1, 1, 1, 1], padding='SAME')
    u_zz = tf.transpose(u_zz, [1, 0, 2, 3]) 
    u_zz = tf.reshape(u_zz, [69, 132, 132])
    
    u_lap = u_xx + u_yy + u_zz
    
    
    # ------------------------equation 2
    temp = v
    v_xx = tf.nn.conv2d(tf.reshape(temp, [69, 132, 132, 1]), filter = xxderi_weights, strides=[1, 1, 1, 1], padding='SAME')
    
    v_xx = tf.reshape(v_xx, [69, 132, 132])
        
    temp = v
    v_yy = tf.nn.conv2d(tf.reshape(temp, [69, 132, 132, 1]), filter = yyderi_weights, strides=[1, 1, 1, 1], padding='SAME')
    
    v_yy = tf.reshape(v_yy, [69, 132, 132])
    
    temp = tf.transpose(v, [1, 0, 2])     
    temp = tf.reshape(temp, [132, 69, 132, 1])     #add transpose
    v_zz = tf.nn.conv2d(temp, filter = zzderi_weights, strides=[1, 1, 1, 1], padding='SAME')
    v_zz = tf.transpose(v_zz, [1, 0, 2, 3]) 
    v_zz = tf.reshape(v_zz, [69, 132, 132])
    
    v_lap = v_xx + v_yy + v_zz
    
    
    # ------------------------equation 3
    temp = w
    w_xx = tf.nn.conv2d(tf.reshape(temp, [69, 132, 132, 1]), filter = xxderi_weights, strides=[1, 1, 1, 1], padding='SAME')
    
    w_xx = tf.reshape(w_xx, [69, 132, 132])
        
    temp = w
    w_yy = tf.nn.conv2d(tf.reshape(temp, [69, 132, 132, 1]), filter = yyderi_weights, strides=[1, 1, 1, 1], padding='SAME')
    
    w_yy = tf.reshape(w_yy, [69, 132, 132])
    
    temp = tf.transpose(w, [1, 0, 2])     
    temp = tf.reshape(temp, [132, 69, 132, 1])     #add transpose
    w_zz = tf.nn.conv2d(temp, filter = zzderi_weights, strides=[1, 1, 1, 1], padding='SAME')
    w_zz = tf.transpose(w_zz, [1, 0, 2, 3]) 
    w_zz = tf.reshape(w_zz, [69, 132, 132])
    
    w_lap = w_xx + w_yy + w_zz
    
    return u_lap, v_lap, w_lap
#perform cnn in x, y, z directions


def calculate_pressure(p, xderi_weights, yderi_weights, zderi_weights):
    
        
    # ------------------------equation 1
    temp = p
    p_x = tf.nn.conv2d(tf.reshape(temp, [69, 132, 132, 1]), filter = xderi_weights, strides=[1, 1, 1, 1], padding='SAME')
    
    p_x = tf.reshape(p_x, [69, 132, 132])
        
    temp = p
    p_y = tf.nn.conv2d(tf.reshape(temp, [69, 132, 132, 1]), filter = yderi_weights, strides=[1, 1, 1, 1], padding='SAME')
    
    p_y = tf.reshape(p_y, [69, 132, 132])
    
    
    temp = tf.transpose(p, [1, 0, 2])     
    temp = tf.reshape(temp, [132, 69, 132, 1])     #add transpose
    p_z = tf.nn.conv2d(temp, filter = zderi_weights, strides=[1, 1, 1, 1], padding='SAME')
    p_z = tf.transpose(p_z, [1, 0, 2, 3]) 
    p_z = tf.reshape(p_z, [69, 132, 132])
    
    
    return p_x, p_y, p_z





def boundary_aug(u, v, w, p):
    
        
    # -----------u_aug
    # temp = tf.zeros([65,2,128])
    temp1 = u[:, 126:128, :]        #low boundary 
    temp2 = u[:, 0:2, :]        #up boundary
    temp3 = u[:, :, 0:2]        #left boundary
    temp4 = u[:, :, 126:128]        #right boundary
    
    temp5 = u[:, 0:2, 0:2]        #up left
    temp6 = u[:, 0:2, 126:128]        #up right
    temp7 = u[:, 126:128, 0:2]        #low left
    temp8 = u[:, 126:128, 126:128]        #low right 
    
    mid = tf.concat([temp3, u, temp4], 2)
    up = tf.concat([temp5, temp2, temp6], 2)
    low = tf.concat([temp7, temp1, temp8], 2)
    comb = tf.concat([up, mid, low], 1)
    
    bottom = comb[63:65, :, :]        #bottom
    top = comb[0:2, :, :]        #top
    
    u_aug = tf.concat([bottom, comb, top], 0)
    
    
    # u_aug[2:66, 2:129, 2:129] = u
    # u_aug[2:66, 2:129, 2:129].assign(u)
    
    # for i in range(2, 66):
    #     u_aug[i, 0:1, 2:129] = u_aug[i, 128:129, 2:129]     #up boundary in middle
    #     u_aug[i, 130:131, 2:129] = u_aug[i, 2:3, 2:129]     #low boundary in middle
    #     u_aug[i, 2:129, 0:1] = u_aug[i, 2:129, 128:129]     #left boundary in middle
    #     u_aug[i, 2:129, 130:131] = u_aug[i, 2:129, 2:3]     #right boundary in middle
        
    #     u_aug[i, 0:1, 0:1] = u_aug[i, 128:129, 128:129]     #up left
    #     u_aug[i, 130:131, 0:1] = u_aug[i, 2:3, 128:129]     #low left
    #     u_aug[i, 0:1, 130:131] = u_aug[i, 128:129, 2:3]     #up right
    #     u_aug[i, 130:131, 130:131] = u_aug[i, 2:3, 2:3]     #low right
        
        
    # u_aug[0:1, :, :] = u_aug[128:129, :, :]     #up boundary, 如果刚好差一个128，就应该是完全相等的     
    # u_aug[130:131, :, :] = u_aug[2:3, :, :]     #low boundary         
    
    # -----------v_aug
    
    temp1 = v[:, 126:128, :]        #low boundary 
    temp2 = v[:, 0:2, :]        #up boundary
    temp3 = v[:, :, 0:2]        #left boundary
    temp4 = v[:, :, 126:128]        #right boundary
    
    temp5 = v[:, 0:2, 0:2]        #up left
    temp6 = v[:, 0:2, 126:128]        #up right
    temp7 = v[:, 126:128, 0:2]        #low left
    temp8 = v[:, 126:128, 126:128]        #low right 
    
    mid = tf.concat([temp3, v, temp4], 2)
    up = tf.concat([temp5, temp2, temp6], 2)
    low = tf.concat([temp7, temp1, temp8], 2)
    comb = tf.concat([up, mid, low], 1)
    
    bottom = comb[63:65, :, :]        #bottom
    top = comb[0:2, :, :]        #top
    
    v_aug = tf.concat([bottom, comb, top], 0)

    # -----------w_aug
    temp1 = w[:, 126:128, :]        #low boundary 
    temp2 = w[:, 0:2, :]        #up boundary
    temp3 = w[:, :, 0:2]        #left boundary
    temp4 = w[:, :, 126:128]        #right boundary
    
    temp5 = w[:, 0:2, 0:2]        #up left
    temp6 = w[:, 0:2, 126:128]        #up right
    temp7 = w[:, 126:128, 0:2]        #low left
    temp8 = w[:, 126:128, 126:128]        #low right 
    
    mid = tf.concat([temp3, w, temp4], 2)
    up = tf.concat([temp5, temp2, temp6], 2)
    low = tf.concat([temp7, temp1, temp8], 2)
    comb = tf.concat([up, mid, low], 1)
    
    bottom = comb[63:65, :, :]        #bottom
    top = comb[0:2, :, :]        #top
    
    w_aug = tf.concat([bottom, comb, top], 0)


    # -----------p_aug
    temp1 = p[:, 126:128, :]        #low boundary 
    temp2 = p[:, 0:2, :]        #up boundary
    temp3 = p[:, :, 0:2]        #left boundary
    temp4 = p[:, :, 126:128]        #right boundary
    
    temp5 = p[:, 0:2, 0:2]        #up left
    temp6 = p[:, 0:2, 126:128]        #up right
    temp7 = p[:, 126:128, 0:2]        #low left
    temp8 = p[:, 126:128, 126:128]        #low right 
    
    mid = tf.concat([temp3, p, temp4], 2)
    up = tf.concat([temp5, temp2, temp6], 2)
    low = tf.concat([temp7, temp1, temp8], 2)
    comb = tf.concat([up, mid, low], 1)
    
    bottom = comb[63:65, :, :]        #bottom
    top = comb[0:2, :, :]        #top
    
    p_aug = tf.concat([bottom, comb, top], 0)
    
    
    return u_aug, v_aug, w_aug, p_aug




def remove_bd(u_aug, v_aug, w_aug):
    
    u = u_aug[2:67, 2:130, 2:130]
    v = v_aug[2:67, 2:130, 2:130]
    w = w_aug[2:67, 2:130, 2:130]

    
    
    return u, v, w

# a = tf.placeholder(tf.float64, [65, 128, 128], name="a")      #a = a_input
u = tf.placeholder(tf.float64, [65, 128, 128], name="u")       #u = u_input
v = tf.placeholder(tf.float64, [65, 128, 128], name="v")       #v = v_input
w = tf.placeholder(tf.float64, [65, 128, 128], name="w")       #w = w_input
p = tf.placeholder(tf.float64, [65, 128, 128], name="p")       #p = p_input

# u_aug = tf.placeholder(tf.float64, [69, 132, 132], name="u_aug")       #u = u_input
# v_aug = tf.placeholder(tf.float64, [69, 132, 132], name="v_aug")       #v = v_input
# w_aug = tf.placeholder(tf.float64, [69, 132, 132], name="w_aug")       #w = w_input
# p_aug = tf.placeholder(tf.float64, [69, 132, 132], name="p_aug")       #p = p_input

mu = tf.placeholder(tf.float64, name="mu")       #mu = mu_input
rho = tf.placeholder(tf.float64, name="rho")       #rho = rho_input
u_targets = tf.placeholder(tf.float64, [65, 128, 128], name="u_targets")       
v_targets = tf.placeholder(tf.float64, [65, 128, 128], name="v_targets")       
w_targets = tf.placeholder(tf.float64, [65, 128, 128], name="w_targets")        
    

learning_rate = tf.placeholder(tf.float64, name="learning_rate")
delta_t = tf.placeholder(tf.float64, name="delta_t")

# inputs = tf.reshape(inputs, [-1, 128, 128, 1])



''' use CNN to approximate spatial derivative  ---- k1  '''


    
#----------------------------- left handside derivative    
num_filters = 1
xderi_weights = create_weights(shape=[3, 3, 1, 1])
yderi_weights = create_weights(shape=[3, 3, 1, 1])
zderi_weights = create_weights(shape=[3, 3, 1, 1])


u_aug, v_aug, w_aug, p_aug = boundary_aug(u, v, w, p)
u_div, v_div, w_div = calculate_div(u_aug, v_aug, w_aug, xderi_weights, yderi_weights, zderi_weights)


#----------------------------- right handside derivative


xxderi_weights = create_weights(shape=[5, 5, 1, 1])  
yyderi_weights = create_weights(shape=[5, 5, 1, 1])  
zzderi_weights = create_weights(shape=[5, 5, 1, 1])  

u_lap, v_lap, w_lap = calculate_laplacian(u_aug, v_aug, w_aug, xxderi_weights, yyderi_weights, zzderi_weights)


#----------------------------- pressure derivative
p_x, p_y, p_z = calculate_pressure(p_aug, xderi_weights, yderi_weights, zderi_weights)

#---------------------remove boundary
u_div, v_div, w_div = remove_bd(u_div, v_div, w_div)
u_lap, v_lap, w_lap = remove_bd(u_lap, v_lap, w_lap)
p_x, p_y, p_z = remove_bd(p_x, p_y, p_z)




# ------------------------update k1
    
f1 = -u_div - p_x/rho + mu*u_lap
f2 = -v_div - p_y/rho + mu*v_lap
f3 = -w_div - p_z/rho + mu*w_lap


    
k1_f1 = f1
k1_f2 = f2
k1_f3 = f3

 



''' use Runge Kutta  '''



# use k1 to calculate k2

u2 = u + delta_t/2 * k1_f1
v2 = v + delta_t/2 * k1_f2
w2 = w + delta_t/2 * k1_f3



u_aug, v_aug, w_aug, p_aug = boundary_aug(u2, v2, w2, p)

u_div, v_div, w_div = calculate_div(u_aug, v_aug, w_aug, xderi_weights, yderi_weights, zderi_weights)


u_lap, v_lap, w_lap = calculate_laplacian(u_aug, v_aug, w_aug, xxderi_weights, yyderi_weights, zzderi_weights)

p_x, p_y, p_z = calculate_pressure(p_aug, xderi_weights, yderi_weights, zderi_weights)


u_div, v_div, w_div = remove_bd(u_div, v_div, w_div)
u_lap, v_lap, w_lap = remove_bd(u_lap, v_lap, w_lap)
p_x, p_y, p_z = remove_bd(p_x, p_y, p_z)

    
f1 = -u_div - p_x/rho + mu*u_lap
f2 = -v_div - p_y/rho + mu*v_lap
f3 = -w_div - p_z/rho + mu*w_lap


    
k2_f1 = f1
k2_f2 = f2
k2_f3 = f3






# use k2 to calculate k3


u3 = u + delta_t/2 * k2_f1
v3 = v + delta_t/2 * k2_f2
w3 = w + delta_t/2 * k2_f3



u_aug, v_aug, w_aug, p_aug = boundary_aug(u3, v3, w3, p)

u_div, v_div, w_div = calculate_div(u_aug, v_aug, w_aug, xderi_weights, yderi_weights, zderi_weights)


u_lap, v_lap, w_lap = calculate_laplacian(u_aug, v_aug, w_aug, xxderi_weights, yyderi_weights, zzderi_weights)

p_x, p_y, p_z = calculate_pressure(p_aug, xderi_weights, yderi_weights, zderi_weights)

u_div, v_div, w_div = remove_bd(u_div, v_div, w_div)
u_lap, v_lap, w_lap = remove_bd(u_lap, v_lap, w_lap)
p_x, p_y, p_z = remove_bd(p_x, p_y, p_z)   
 

f1 = -u_div - p_x/rho + mu*u_lap
f2 = -v_div - p_y/rho + mu*v_lap
f3 = -w_div - p_z/rho + mu*w_lap


    
k3_f1 = f1
k3_f2 = f2
k3_f3 = f3






# use k3 to calculate k4


u4 = u + delta_t * k3_f1
v4 = v + delta_t * k3_f2
w4 = w + delta_t * k3_f3



u_aug, v_aug, w_aug, p_aug = boundary_aug(u4, v4, w4, p)

u_div, v_div, w_div = calculate_div(u_aug, v_aug, w_aug, xderi_weights, yderi_weights, zderi_weights)


u_lap, v_lap, w_lap = calculate_laplacian(u_aug, v_aug, w_aug, xxderi_weights, yyderi_weights, zzderi_weights)

p_x, p_y, p_z = calculate_pressure(p_aug, xderi_weights, yderi_weights, zderi_weights)
    
u_div, v_div, w_div = remove_bd(u_div, v_div, w_div)
u_lap, v_lap, w_lap = remove_bd(u_lap, v_lap, w_lap)
p_x, p_y, p_z = remove_bd(p_x, p_y, p_z)   
 

f1 = -u_div - p_x/rho + mu*u_lap
f2 = -v_div - p_y/rho + mu*v_lap
f3 = -w_div - p_z/rho + mu*w_lap


    
k4_f1 = f1
k4_f2 = f2
k4_f3 = f3




# add weights， train weights, need more weights??    
w1 = tf.Variable(tf.constant(1, dtype=tf.dtypes.float64))
w2 = tf.Variable(tf.constant(2, dtype=tf.dtypes.float64))
w3 = tf.Variable(tf.constant(2, dtype=tf.dtypes.float64))
w4 = tf.Variable(tf.constant(1, dtype=tf.dtypes.float64))

w5 = tf.Variable(tf.constant(1, dtype=tf.dtypes.float64))
w6 = tf.Variable(tf.constant(2, dtype=tf.dtypes.float64))
w7 = tf.Variable(tf.constant(2, dtype=tf.dtypes.float64))
w8 = tf.Variable(tf.constant(1, dtype=tf.dtypes.float64))

w9 = tf.Variable(tf.constant(1, dtype=tf.dtypes.float64))
w10 = tf.Variable(tf.constant(2, dtype=tf.dtypes.float64))
w11 = tf.Variable(tf.constant(2, dtype=tf.dtypes.float64))
w12 = tf.Variable(tf.constant(1, dtype=tf.dtypes.float64))

u_new = u + 1/6 * delta_t * (w1*k1_f1 + w2*k2_f1 + w3*k3_f1 + w4*k4_f1)    
v_new = v + 1/6 * delta_t * (w5*k2_f2 + w6*k2_f2 + w7*k3_f2 + w8*k4_f2)    
w_new = w + 1/6 * delta_t * (w9*k1_f3 + w10*k2_f3 + w11*k3_f3 + w12*k4_f3)     



#define loss



# cost = tf.reduce_sum(tf.sqrt(tf.reshape(targets - a_new, [-1])/(65*128*128)))


cost_1 = tf.sqrt(tf.reduce_sum(tf.square(tf.reshape(u_targets - u_new, [-1])))
               /(65*128*128))

cost_2 = tf.sqrt(tf.reduce_sum(tf.square(tf.reshape(v_targets - v_new, [-1])))
               /(65*128*128))

cost_3 = tf.sqrt(tf.reduce_sum(tf.square(tf.reshape(w_targets - w_new, [-1])))
               /(65*128*128))

cost = (cost_1 + cost_2 + cost_3)/3  
     
optimizer = tf.train.AdamOptimizer(learning_rate)
gradients = optimizer.compute_gradients(cost)

train_op = optimizer.apply_gradients(gradients)



saver = tf.train.Saver(max_to_keep=5)


'''    Load data and training   '''


mu_input = 0.025
rho_input = 1.0
lr = 0.01
delta_t_input = 1      
epochs = 50
# data_length = 9         #the last index of data is only for testing, so 6 - 1 = 5 overall data
# train_length = 5        #try 5, 6 for 9 time slots
data_length = 20         #the last index of data is only for testing, so 6 - 1 = 5 overall data
train_length = 10        #try 5, 6 for 9 time slots
test_length = data_length - train_length
# train_length = 

for i in range(99, 99 + data_length):           #load data
    u_input = 'u_input_' + str(i)
    v_input = 'v_input_' + str(i)
    w_input = 'w_input_' + str(i)
    p_input = 'p_input_' + str(i)
    
    data_u = np.load('img_u_dns' + str(i) + '.npy')
    data_v = np.load('img_v_dns' + str(i) + '.npy')
    data_w = np.load('img_w_dns' + str(i) + '.npy')
    data_p = np.load('img_p_dns' + str(i) + '.npy')
    
    maximum0_dns = data_u.max()
    minimum0_dns = data_u.min()
    data_u = (data_u - minimum0_dns)/(maximum0_dns - minimum0_dns)
    
    maximum1_dns = data_v.max()
    minimum1_dns = data_v.min()
    data_v = (data_v - minimum0_dns)/(maximum0_dns - minimum0_dns)
    
    maximum2_dns = data_w.max()
    minimum2_dns = data_w.min()
    data_w = (data_w - minimum0_dns)/(maximum0_dns - minimum0_dns)
    
    maximum3_dns = data_p.max()
    minimum3_dns = data_p.min()
    data_p = (data_p - minimum0_dns)/(maximum0_dns - minimum0_dns)
    
                
    vars()[u_input] = data_u
    vars()[v_input] = data_v
    vars()[w_input] = data_w
    vars()[p_input] = data_p
    
    
# u_input_99 = np.load('img_u_dns99.npy')
# v_input_99 = np.load('img_v_dns99.npy')
# w_input_99 = np.load('img_w_dns99.npy')
# features_99 = np.load('img_a_dns99.npy')
# # q = np.load('img_q_dns99.npy')


# u_input_100 = np.load('img_u_dns100.npy')
# v_input_100 = np.load('img_v_dns100.npy')
# w_input_100 = np.load('img_w_dns100.npy')
# features_100 = np.load('img_a_dns100.npy')




# u_input_101 = np.load('img_u_dns101.npy')
# v_input_101 = np.load('img_v_dns101.npy')
# w_input_101 = np.load('img_w_dns101.npy')
# features_101 = np.load('img_a_dns101.npy')


# u_input_102 = np.load('img_u_dns102.npy')
# v_input_102 = np.load('img_v_dns102.npy')
# w_input_102 = np.load('img_w_dns102.npy')
# features_102 = np.load('img_a_dns102.npy')


# u_input_103 = np.load('img_u_dns103.npy')
# v_input_103 = np.load('img_v_dns103.npy')
# w_input_103 = np.load('img_w_dns103.npy')
# features_103 = np.load('img_a_dns103.npy')


#saver.save(sess,'./models/STAN_reservoir_pretrain_old.ckpt') 

# saver.restore(sess,'./models/STAN_reservoir_pretrain_old.ckpt') 

# train_graph = tf.Graph()
sess = tf.Session()
sess.run(tf.global_variables_initializer())




for i in range(epochs):                 #optimizae same sets of weights


        for k in range(99, 99 + train_length):
    
            _, loss, u_pred, v_pred, w_pred = sess.run(
              [train_op, cost, u_new, v_new, w_new],
              {u: eval('u_input_' + str(k)),
              v: eval('v_input_' + str(k)),
              w: eval('w_input_' + str(k)),
              p: eval('p_input_' + str(k)),  #use real values to start
              u_targets: eval('u_input_' + str(k + 1)),
              v_targets: eval('v_input_' + str(k + 1)),
              w_targets: eval('w_input_' + str(k + 1)),
              learning_rate: lr,
              delta_t: delta_t_input,
              mu: mu_input,
              rho: rho_input})      #need to specify rho
            
            
        u_pred, v_pred, w_pred = sess.run(      #use real value at the beginning of prediction
            [u_new, v_new, w_new],
            {u: eval('u_input_' + str(99 + train_length)),
             v: eval('v_input_' + str(99 + train_length)),
             w: eval('w_input_' + str(99 + train_length)),
             p: eval('p_input_' + str(99 + train_length)),
             u_targets: eval('u_input_' + str(99 + train_length + 1)),
             v_targets: eval('v_input_' + str(99 + train_length + 1)),
             w_targets: eval('w_input_' + str(99 + train_length + 1)),
             learning_rate: lr,
             delta_t: delta_t_input,
             mu: mu_input,
             rho: rho_input}
            )  
            
        for l in range(99 + train_length + 1, 99 + data_length - 1):    # the last index is for testing, 99 + data_length - 1 is the last index of data
            # print(l)
            loss, u_pred, v_pred, w_pred = sess.run(       
                [cost, u_new, v_new, w_new],
                {u: u_pred,
                 v: v_pred,
                 w: w_pred,
                 p: eval('p_input_' + str(l)),
                 u_targets: eval('u_input_' + str(l + 1)),
                 v_targets: eval('v_input_' + str(l + 1)),
                 w_targets: eval('w_input_' + str(l + 1)),
                 learning_rate: lr,
                 delta_t: delta_t_input,
                 mu: mu_input,
                 rho: rho_input})             

        # loss = np.sqrt(np.sum(np.square(np.reshape(u_pred - eval('u_input_' + str(99 + data_length - 1)), 
        #                                            [-1])))
        #                    /(65*128*128)) + np.sqrt(np.sum(np.square(np.reshape(v_pred - eval('v_input_' + str(99 + data_length - 1)), 
        #                                            [-1])))
        #                    /(65*128*128)) + np.sqrt(np.sum(np.square(np.reshape(w_pred - eval('w_input_' + str(99 + data_length - 1)), 
        #                                            [-1])))
        #                    /(65*128*128))
         
         
        print('Epoch '+str(i)+': loss at 118 '+"{:.6f}".format(loss)) 

                    
         

saver.save(sess,'./periodic bc model/RK-vector_50_118_10.ckpt') 

# saver.save(sess,'./normalized model/RK-vector_10_118_10.ckpt')   

# plt.imshow(features_100[5,:,:])
# plt.show()
# plt.imshow(features_100[55,:,:])
# plt.show()

# plt.imshow(pred[5,:,:])
# plt.show()
# plt.imshow(pred[55,:,:])
# plt.show()


# plt.imshow(features_input_107[5,:,:])
# plt.show()
# plt.imshow(features_input_107[55,:,:])
# plt.show()


# pred_101 = np.array(pred_101)
# pred_101 = np.reshape(pred_101, [65, 128, 128])

#--------------------------------------------
plt.imshow(u_pred[5,:,:])
plt.show()
plt.imshow(u_pred[55,:,:])
plt.show()

# plt.imshow(v_pred[5,:,:])
# plt.show()
# plt.imshow(v_pred[55,:,:])
# plt.show()

# plt.imshow(w_pred[5,:,:])
# plt.show()
# plt.imshow(w_pred[55,:,:])
# plt.show()


# plt.imshow(u_input_118[5,:,:])
# plt.show()
# plt.imshow(u_input_118[55,:,:])
# plt.show()


# plt.imshow(v_input_111[5,:,:])
# plt.show()
# plt.imshow(v_input_111[55,:,:])
# plt.show()

#--------------------------------------------

# plt.imshow(v_104[5,:,:])
# plt.show()
# plt.imshow(v_104[55,:,:])
# plt.show()

# plt.imshow(w_104[5,:,:])
# plt.show()
# plt.imshow(w_104[55,:,:])
# plt.show()


# # pred_101 = np.array(pred_101)
# # pred_101 = np.reshape(pred_101, [65, 128, 128])
# plt.imshow(pred_104[5,:,:])
# plt.show()
# plt.imshow(pred_104[55,:,:])
# plt.show()


# plot(a_new[5,:,:])






