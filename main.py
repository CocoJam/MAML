import numpy as np
import random
import tensorflow as tf
from tensorflow.python.platform import flags
from sin_wave_generation import sinGen
Flags = flags.FLAGS

flags.DEFINE_string('datasource', 'sinusoid', 'sinusoid')
flags.DEFINE_integer('metatrain_iterations', 15000, 'number of metatraining iterations.')
flags.DEFINE_integer('meta_batch_size', 25, 'number of tasks sampled per meta-update')
flags.DEFINE_float('meta_lr', 0.001, 'the base learning rate of the generator')
flags.DEFINE_integer('update_batch_size', 5, 'number of examples used for inner gradient update (K for K-shot learning).')
flags.DEFINE_float('update_lr', 1e-3, 'step size alpha for inner gradient update.') 
flags.DEFINE_bool('train', True, 'True to train, False to test.')

flags.DEFINE_string('norm', 'batch_norm', 'batch_norm, layer_norm, or None')

def main():
    if (Flags.datasource == "sinusoid"):
        dataGen = sinGen(Flags.update_batch_size*2, Flags.meta_batch_size)
