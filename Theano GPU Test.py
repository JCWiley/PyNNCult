import os
#os.environ["MKL_THREADING_LAYER"] = "GNU"
import tensorflow as tf
os.environ['THEANO_FLAGS'] = "device=opencl,force_device=True,floatX=float32"
print(os.path.expanduser('~'))

from keras import backend as K
print(K.tensorflow_backend._get_available_gpus())
print(tf.__version__)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

print(sess)