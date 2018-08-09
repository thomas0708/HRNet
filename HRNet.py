# coding: utf-8

# HRNet: holographic reconstruction network
# The goal of this notebook is to implement HRNet for digital holographic reconstruction.

# These are all the modules we'll be using later. Make sure you can import them before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import scipy.ndimage as ndimage
from scipy.misc import imresize, imsave, imread
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math, cmath
from scipy import signal, io
import h5py
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import cv2
import os
import time
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_mse as mse
from skimage.measure import compare_nrmse as nrmse
from skimage.measure import compare_psnr as psnr
get_ipython().magic('matplotlib inline')


def normalize_image(img, NewMin, NewMax):
#     return (img-np.amin(img))/(np.amax(img)-np.amin(img)) # [0,1]
    return (img-np.amin(img))*(NewMax-NewMin)/(np.amax(img)-np.amin(img))+NewMin

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(M, N), cmap='gray', interpolation='none')

    return fig

def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' Gaussian MATLAB function"""
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    
    return g / tf.reduce_sum(g)


def tf_ssim(img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5):
    window = _tf_fspecial_gauss(size, sigma) # window shape [size, size]
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1],padding='VALID')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1],padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1],padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1],padding='VALID') - mu1_mu2
    if cs_map:
        value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)),
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))

    if mean_metric:
        value = tf.reduce_mean(value)
        
    return value


def tf_ms_ssim(img1, img2, mean_metric=True, level=5):
    weight = tf.constant([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=tf.float32)
    mssim = []
    mcs = []
    for l in range(level):
        ssim_map, cs_map = tf_ssim(img1, img2, cs_map=True, mean_metric=False)
        mssim.append(tf.reduce_mean(ssim_map))
        mcs.append(tf.reduce_mean(cs_map))
        filtered_im1 = tf.nn.avg_pool(img1, [1,2,2,1], [1,2,2,1], padding='SAME')
        filtered_im2 = tf.nn.avg_pool(img2, [1,2,2,1], [1,2,2,1], padding='SAME')
        img1 = filtered_im1
        img2 = filtered_im2

    # list to tensor of dim D+1
    mssim = tf.stack(mssim, axis=0)
    mcs = tf.stack(mcs, axis=0)

    value = (tf.reduce_prod(mcs[0:level-1]**weight[0:level-1])*
                            (mssim[level-1]**weight[level-1]))

    if mean_metric:
        value = tf.reduce_mean(value)
        
    return value


mat = h5py.File('../holo_data.mat')
print(type(mat), mat.keys())

dataset = mat['dataset'][()]
labels = mat['labels'][()]

dataset = np.transpose(dataset)
labels = np.transpose(labels)

del mat

print(type(dataset), dataset.shape)
print(type(labels), labels.shape)

M = dataset.shape[1]
N = dataset.shape[2]

print(M, N)


### Data normalization

print(np.amax(dataset), np.amin(dataset))
print(np.amax(labels), np.amin(labels))

dataset = normalize_image(dataset, 0, 1)
labels = normalize_image(labels, 0, 1)

print(np.amax(dataset), np.amin(dataset))
print(np.amax(labels), np.amin(labels))


train_dataset, test_dataset, train_labels, test_labels = train_test_split(
    dataset, labels, test_size=0.2, random_state=42)

print('Training set', train_dataset.shape, train_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


fig = plt.figure()
fig.add_subplot(121)
plt.imshow(train_dataset[0], cmap='gray'), plt.axis('off')
fig.add_subplot(122)
plt.imshow(test_dataset[0], cmap='gray'), plt.axis('off')

fig = plot(train_dataset[0:16])
fig = plot(train_labels[0:16])


NUM_TRAIN = train_dataset.shape[0]
NUM_TEST = test_dataset.shape[0]


### Data pre-processing: zero-centred and scaled (unit variance)

mean_train = np.mean(train_dataset)
mean_test = np.mean(test_dataset)
print(mean_train, mean_test)

std_train = np.std(train_dataset)
std_test = np.std(test_dataset)
print(std_train, std_test)

train_dataset = (train_dataset - mean_train) / std_train
test_dataset = (test_dataset - mean_test) / std_test


### Reformat into a TensorFlow-friendly shape:
# - convolutions need the image data formatted as a cube (width by height by #channels)
# - labels as float 1-hot encodings.

image_size = N
num_labels_gt = image_size * image_size
num_channels = 1

def reformat_data(dataset):
  return dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)

def reformat_gt(labels):
  return labels.reshape((-1, num_labels_gt)).astype(np.float32)

def reformat_gt2(labels):
  return labels.reshape((-1, image_size, image_size)).astype(np.float32)

print('Before reformat')
print('Training set', train_dataset.shape, train_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

train_dataset = reformat_data(train_dataset)
test_dataset = reformat_data(test_dataset)

train_labels = reformat_data(train_labels)
test_labels = reformat_data(test_labels)

print('After reformat')
print('Training set', train_dataset.shape, train_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


fig = plt.figure()
fig.add_subplot(221)
plt.imshow(train_dataset[0,:,:,0], cmap='gray'), plt.axis('off')
fig.add_subplot(222)
plt.imshow(test_dataset[0,:,:,0], cmap='gray'), plt.axis('off')
fig.add_subplot(223)
plt.imshow(train_labels[0,:,:,0], cmap='gray'), plt.axis('off')
fig.add_subplot(224)
plt.imshow(test_labels[0,:,:,0], cmap='gray'), plt.axis('off')


### HRNet construction

def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    
    return tf.Variable(initial, name=name)

def bias_variable(shape, name=None):
    initial = tf.constant(1.0, shape=shape.get_shape().as_list()[-1])
    
    return tf.Variable(initial, name=name)

def softmax_layer(inpt, shape):
    fc_w = weight_variable(shape)
    fc_b = tf.Variable(tf.zeros([shape[1]]))

    fc_h = tf.nn.softmax(tf.matmul(inpt, fc_w) + fc_b)

    return fc_h

def softmax_layer2(inpt, shape):
    fc_w = weight_variable(shape)
    fc_b = tf.Variable(tf.zeros([shape[1]]))

    fc_h = tf.matmul(inpt, fc_w) + fc_b

    return fc_h

def sigmoid_layer(inpt, shape):
    fc_w = weight_variable(shape)
    fc_b = tf.Variable(tf.zeros([shape[1]]))

    fc_h = tf.nn.sigmoid(tf.matmul(inpt, fc_w) + fc_b)

    return fc_h

def fcn_layer(inpt, shape): # fully-convolutional layer
    fcn_w = weight_variable(shape)
    fcn_b = tf.Variable(tf.zeros([shape[1]]))

    fcn_h = tf.matmul(inpt, fcn_w) + fcn_b

    mean, var = tf.nn.moments(fcn_h, axes=[0,1])
    beta = tf.Variable(tf.zeros([shape[1]]), name="beta")
    gamma = weight_variable([shape[1]], name="gamma")
    
    batch_norm = tf.nn.batch_norm_with_global_normalization(
        fcn_h, mean, var, beta, gamma, 0.001,
        scale_after_normalization=True)
    
    out = tf.nn.relu(batch_norm)

    return out

def batch_norm_layer(inpt):
    """Batch normalization for a 4-D tensor"""
    assert len(inpt.get_shape()) == 4
    filter_shape = inpt.get_shape().as_list()
    mean, var = tf.nn.moments(inpt, axes=[0, 1, 2])
    out_channels = filter_shape[3]
    offset = tf.Variable(tf.zeros([out_channels]))
    scale = tf.Variable(tf.ones([out_channels]))
    batch_norm = tf.nn.batch_normalization(inpt, mean, var, offset, scale, 0.001)
    
    return batch_norm

def conv_layer(inpt, filter_shape, stride):
    out_channels = filter_shape[3]

    filter_ = weight_variable(filter_shape)
    conv = tf.nn.conv2d(inpt, filter=filter_, strides=[1, stride, stride, 1], padding="SAME")

    """
    mean, var = tf.nn.moments(conv, axes=[0,1,2])
    beta = tf.Variable(tf.zeros([out_channels]), name="beta")
    gamma = weight_variable([out_channels], name="gamma")
    
    batch_norm = tf.nn.batch_norm_with_global_normalization(
        conv, mean, var, beta, gamma, 0.001,
        scale_after_normalization=True)
    """

    batch_norm = batch_norm_layer(conv)
    out = tf.nn.relu(batch_norm)

    return out

def residual_block(inpt, output_depth, down_sample, projection=False):
    input_depth = inpt.get_shape().as_list()[3]
    if down_sample:
        filter_ = [1,2,2,1]
        inpt = tf.nn.max_pool(inpt, ksize=filter_, strides=filter_, padding='SAME')

    conv1 = conv_layer(inpt, [3, 3, input_depth, output_depth], 1)
    conv2 = conv_layer(conv1, [3, 3, output_depth, output_depth], 1)

    if input_depth != output_depth:
        if projection:
            """Option B: Projection shortcut"""
            input_layer = conv_layer(inpt, [1, 1, input_depth, output_depth], 2)
        else:
            """Option A: Zero-padding"""
            input_layer = tf.pad(inpt, [[0,0], [0,0], [0,0], [0, output_depth - input_depth]])
    else:
        input_layer = inpt

    res = conv2 + input_layer
    
    return res

def _phase_shift(I, r):
    """Helper function with main phase shift operation"""
    bsize, a, b, c = I.get_shape().as_list()
    X = tf.reshape(I, (bsize, a, b, r, r))
    X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
    X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
    X = tf.concat([tf.squeeze(x) for x in X], 2)  # bsize, b, a*r, r
    X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
    X = tf.concat([tf.squeeze(x) for x in X], 2)  # bsize, a*r, b*r
    
    print('X:', X.get_shape())
    return tf.reshape(X, (bsize, a*r, b*r, 1))


def resnet(inpt, n):
    if n < 20 or (n - 20) % 12 != 0:
        print("ResNet depth invalid.")
        return

    num_conv = (n - 20) / 12 + 1
    layers = []

    with tf.variable_scope('conv1'):
        conv1 = conv_layer(inpt, [3, 3, 1, 32], 1)
        print('inpt:', inpt.get_shape())
        print('conv1:', conv1.get_shape())
        layers.append(conv1)

    for i in range(num_conv):
        down_sample = True if i == 0 else False
        with tf.variable_scope('conv2_%d' % (i+1)):
            conv2_x = residual_block(layers[-1], 64, down_sample)
            print('conv2_x:', conv2_x.get_shape())
            conv2 = residual_block(conv2_x, 64, False)
            print('conv2:', conv2.get_shape())
            layers.append(conv2_x)
            layers.append(conv2)

        assert conv2.get_shape().as_list()[1:] == [M/2, N/2, 64]

    for i in range(num_conv):
        down_sample = True if i == 0 else False
        with tf.variable_scope('conv3_%d' % (i+1)):
            conv3_x = residual_block(layers[-1], 128, down_sample)
            print('conv3_x:', conv3_x.get_shape())
            conv3 = residual_block(conv3_x, 128, False)
            print('conv3:', conv3.get_shape())
            layers.append(conv3_x)
            layers.append(conv3)

        assert conv3.get_shape().as_list()[1:] == [M/4, N/4, 128]
    
    for i in range(num_conv):
        down_sample = True if i == 0 else False
        with tf.variable_scope('conv4_%d' % (i+1)):
            conv4_x = residual_block(layers[-1], 256, down_sample)
            print('conv4_x:', conv4_x.get_shape())
            conv4 = residual_block(conv4_x, 256, False)
            print('conv4:', conv4.get_shape())
            layers.append(conv4_x)
            layers.append(conv4)

        assert conv4.get_shape().as_list()[1:] == [M/8, N/8, 256]

    with tf.variable_scope('fc'):
        #out = fcn_layer(global_pool, [256, 32*32*4*4]) # image_size*image_size
        conv5 = conv_layer(layers[-1], [3, 3, 256, 64], 1)
        print('conv5:', conv5.get_shape())
        out = _phase_shift(conv5, 8)
        print('out:', out.get_shape())
        layers.append(out)

    return layers[-1]


X_train = train_dataset
X_test = test_dataset

Y_train = train_labels
Y_test = test_labels


batch_size = 10

X = tf.placeholder("float", [batch_size, N, N, 1])
Y = tf.placeholder("float", [batch_size, N, N, 1])
learning_rate = tf.placeholder("float", [])

### HRNet model

n = 1
layer = 20 + n*12
logits = resnet(X, layer)

logits = tf.reshape(logits, [-1, N, N, 1])
print(logits.get_shape(), Y.get_shape())


loss = tf.reduce_mean(tf.nn.l2_loss(tf.subtract(logits, Y))) # mse/L2

opt = tf.train.AdamOptimizer(learning_rate, 0.9)
train_op = opt.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


### Initial train

total_loss = []

for epoch in range (1):
    print('Training at epoch %d' % epoch)
    for i in range (0, NUM_TRAIN, batch_size):
        feed_dict={
            X: X_train[i : i + batch_size],
            Y: Y_train[i : i + batch_size],
            learning_rate: 0.01}
        _, l, pred = sess.run([train_op, loss, logits], feed_dict=feed_dict)
        total_loss.append(l)
        if i % 1 == 0:
            print('Minibatch loss at step %d: %f' % (i, l))
    print('Minibatch loss at step %d: %f' % (i + batch_size, l))
    
plt.plot(total_loss)


### Train again

for epoch in range (24):
    print('Training at epoch %d' % epoch)
    for i in range (0, NUM_TRAIN, batch_size):
        feed_dict={
            X: X_train[i : i + batch_size], 
            Y: Y_train[i : i + batch_size],
            learning_rate: 0.01}
        _, l, pred = sess.run([train_op, loss, logits], feed_dict=feed_dict)
        total_loss.append(l)
        if i % 100 == 0:
            print('Minibatch loss at step %d: %f' % (i, l))
    print('Minibatch loss at step %d: %f' % (i + batch_size, l))
    
plt.plot(total_loss)


### Save trained model

model_directory = './models'

saver = tf.train.Saver()

if not os.path.exists(model_directory):
    os.makedirs(model_directory)
saver.save(sess, model_directory+'/model-'+str(i)+'.cptk')
print('Saved Model')


### Test

t = time.time()

for i in range (0, NUM_TEST, batch_size):
    feed_dict={
        X: X_test[i : i + batch_size], 
        Y: Y_test[i : i + batch_size],
        learning_rate: 0.001}
    _, pred = sess.run([train_op, logits], feed_dict=feed_dict)
    if i % 100 == 0:
        print("testing at image #%d" % i)
print("testing at image #%d" % (i + batch_size))

elapsed = time.time() - t
print(elapsed)


### Display and save image

pred_imgs = pred.reshape((batch_size, image_size, image_size, 1)).astype(np.float32)

img_start = 0
fig = plot(pred_imgs[-17 : -1, :, :])

# plt.savefig('test_pred_exp.eps', bbox_inches='tight', pad_inches=0)


### Compute PSNR and SSIM

test_psnr = 0
test_ssim = 0
for i in range (0, NUM_TEST):
    test_psnr = test_psnr + psnr(pred_imgs[i, :, :], Y_test[i, :, :])
    test_ssim = test_ssim + ssim(pred_imgs[i, :, :], Y_test[i, :, :])

test_psnr = test_psnr / NUM_TEST
test_ssim = test_ssim / NUM_TEST

print("testing PSNR: %f" % (test_psnr))
print("testing SSIM: %f" % (test_ssim))

sess.close()