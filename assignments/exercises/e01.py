"""
Simple TensorFlow exercises
You should thoroughly test your code
"""

#test

import tensorflow as tf

###############################################################################
# 1a: Create two random 0-d tensors x and y of any distribution.
# Create a TensorFlow object that returns x + y if x > y, and x - y otherwise.
# Hint: look up tf.cond()
# I do the first problem for you
###############################################################################

ax = tf.random_uniform([])  # Empty array as shape creates a scalar.
ay = tf.random_uniform([])
aout = tf.cond(tf.greater(ax, ay), lambda: tf.add(ax, ay), lambda: tf.subtract(ax, ay))

#with tf.Session() as sess:
#	print(sess.run(aout))

###############################################################################
# 1b: Create two 0-d tensors x and y randomly selected from the range [-1, 1).
# Return x + y if x < y, x - y if x > y, 0 otherwise.
# Hint: sook up tf.case().
###############################################################################

bx = tf.Variable(tf.random_uniform([], minval=-1.0, maxval=1.0))
by = tf.Variable(tf.random_uniform([], minval=-1.0, maxval=1.0))

def badd(): return tf.add(bx,by)
def bsubtract(): return tf.subtract(bx,by)
def bdefault(): return tf.constant(0.0)

bout = tf.case({tf.less(bx, by): badd, tf.greater(bx, by): bsubtract}, default=bdefault, exclusive=True)

#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#    print(sess.run(bx))
#    print(sess.run(by))
#    print(sess.run(bout))

###############################################################################
# 1c: Create the tensor x of the value [[0, -2, -1], [0, 1, 2]] 
# and y as a tensor of zeros with the same shape as x.
# Return a boolean tensor that yields Trues if x equals y element-wise.
# Hint: Look up tf.equal().
###############################################################################

cx = [[0, -2, -1], [0, 1, 2]]
cy = [[0, 0, 0], [0, 0, 0]]

cout = tf.equal(cx, cy)

#with tf.Session() as sess:
#    print(sess.run(cout))

###############################################################################
# 1d: Create the tensor x of value 
# [29.05088806,  27.61298943,  31.19073486,  29.35532951,
#  30.97266006,  26.67541885,  38.08450317,  20.74983215,
#  34.94445419,  34.45999146,  29.06485367,  36.01657104,
#  27.88236427,  20.56035233,  30.20379066,  29.51215172,
#  33.71149445,  28.59134293,  36.05556488,  28.66994858].
# Get the indices of elements in x whose values are greater than 30.
# Hint: Use tf.where().
# Then extract elements whose values are greater than 30.
# Hint: Use tf.gather().
###############################################################################

dx = [29.05088806,  27.61298943,  31.19073486,  29.35532951,
  30.97266006,  26.67541885,  38.08450317,  20.74983215,
  34.94445419,  34.45999146,  29.06485367,  36.01657104,
  27.88236427,  20.56035233,  30.20379066,  29.51215172,
  33.71149445,  28.59134293,  36.05556488,  28.66994858]

dy = [30]

dwhere = tf.where(tf.greater(dx, dy))

dgather = tf.gather(dx, dwhere)

#with tf.Session() as sess:
#	print(sess.run(dwhere))
#	print(sess.run(dgather))

###############################################################################
# 1e: Create a diagnoal 2-d tensor of size 6 x 6 with the diagonal values of 1,
# 2, ..., 6
# Hint: Use tf.range() and tf.diag().
###############################################################################

ex = tf.range(1, 7, 1)
eout = tf.diag(ex)

#with tf.Session() as sess:
#	print(sess.run(eout))

###############################################################################
# 1f: Create a random 2-d tensor of size 10 x 10 from any distribution.
# Calculate its determinant.
# Hint: Look at tf.matrix_determinant().
###############################################################################

fx = tf.Variable(tf.random_normal([10,10], mean=10, stddev=1))
fout = tf.matrix_determinant(fx)

#with tf.Session() as sess:
#	sess.run(tf.global_variables_initializer())
#	print(sess.run(fx))
#	print(sess.run(fout))


###############################################################################
# 1g: Create tensor x with value [5, 2, 3, 5, 10, 6, 2, 3, 4, 2, 1, 1, 0, 9].
# Return the unique elements in x
# Hint: use tf.unique(). Keep in mind that tf.unique() returns a tuple.
###############################################################################

gx = [5, 2, 3, 5, 10, 6, 2, 3, 4, 2, 1, 1, 0, 9]
gy = tf.unique(gx)
gout = gy.y

#with tf.Session() as sess:
#	print(sess.run(gout))

###############################################################################
# 1h: Create two tensors x and y of shape 300 from any normal distribution,
# as long as they are from the same distribution.
# Use tf.cond() to return:
# - The mean squared error of (x - y) if the average of all elements in (x - y)
#   is negative, or
# - The sum of absolute value of all elements in the tensor (x - y) otherwise.
# Hint: see the Huber loss function in the lecture slides 3.
###############################################################################

hx = tf.Variable(tf.random_normal([300], mean=10, stddev=1))
hy = tf.Variable(tf.random_normal([300], mean=10, stddev=1))

hsub = tf.subtract(hx, hy)

hmean = tf.reduce_mean(hsub)

def hf1(): return tf.reduce_mean(tf.square(hsub))
def hf2(): return tf.reduce_sum(tf.abs(hsub))

hout = tf.cond(tf.less(hmean, 0), hf1, hf2)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print(sess.run(hx))
	print(sess.run(hy))
	print(sess.run(hout))
