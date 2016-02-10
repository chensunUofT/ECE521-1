import tensorflow as tf
import math

def gaussian(x, mean, std):
  result = tf.sub(x, mean)
  result = tf.mul(result,tf.inv(std))
  result = tf.exp(-tf.square(result)/2)
  return tf.mul(result,tf.inv(std*tf.sqrt(math.pi * 2)))

def log_likelihood(out_pi, out_sigma, out_mu, y):
  result = gaussian(y, out_mu, out_sigma)
  result = tf.mul(result, out_pi)
  result = tf.reduce_sum(result, 1, keep_dims=True)
  result = -tf.log(result)
  return tf.reduce_mean(result)