import tensorflow as tf
import numpy as np
from tensorflow.python.platform import flags
Flags = flags.FLAGS
class MAML:
    def __init__(self, inputTensor= None ,inputShape = 1, outShape = 1):
        self.inputShape = inputShape
        self.outShape = outShape
        self.adamlr = Flags.update_lr
        self.hidden = [40,40]

    def placeholderInit(self, inputTensor = None):
        if inputTensor is None:
            self.input_1 = tf.placeholder(tf.float32)
            self.input_2 = tf.placeholder(tf.float32)
            self.label_1 = tf.placeholder(tf.float32)
            self.label_2 = tf.placeholder(tf.float32)
        else:
            self.input_1 = inputTensor["input_1"]
            self.input_2 = inputTensor["input_2"]
            self.label_1 = inputTensor["label_1"]
            self.label_2 = inputTensor["label_2"]

    def weightInit(self):
        weights = {}
        weights["w1"] = tf.Variable(tf.truncated_normal([self.inputShape, self.hidden[0]], stddev=0.01))
        weights["b1"] = tf.Variable(tf.zeros([self.hidden[0]]))
        for i in range(1, self.hidden):
            weights["w"+ str(i+1)] = tf.Variable(tf.truncated_normal([self.hidden[i-1], self.hidden[i]], stddev=0.01))
            weights["b"+ str(i+1)] = tf.Variable(tf.zeros([self.hidden[i]]))
        weights['w'+str(len(self.hidden)+1)] = tf.Variable(tf.truncated_normal([self.hidden[-1], self.outShape], stddev=0.01))
        weights['b'+str(len(self.hidden)+1)] = tf.Variable(tf.zeros([self.outShape]))
        return weights
    
    def normalization(self, tensors, activationFunc, reuse,scopeName):
        if(Flags.norm == "batch_norm"):
            tf.contrib.layers.batch_norm(tensors, activation_fn = activationFunc, reuse= reuse, scope = scopeName)
        elif(Flags.norm == "layer_norm"):
            tf.contrib.layers.layer_norm(tensors, activation_fn = activationFunc, reuse= reuse, scope = scopeName)
        else:
            if activationFunc is not None:
                return activationFunc(tensors)
            else:
                return tensors
    
    def forwardStep(self, inputTensor, weight ,reuse=False):
        hidden = tf.add(tf.matmul(inputTensor, weight["w1"]), weight["b1"])
        hidden = self.normalization(hidden, tf.nn.relu, reuse= reuse, scopeName = "0")
        for i in range(1, len(self.hidden)):
            hidden = tf.add(tf.matmul(hidden, weight["w"+ str(i+1)]), weight["b"+ str(i+1)])
            hidden = self.normalization(hidden, tf.nn.relu, reuse= reuse, scopeName = str(i))
        return  tf.add(tf.matmul(hidden, weight["w"+str(len(self.hidden))+1]), weight["b"+ str(len(self.hidden)+1)])
    
    def lossfunction(self, pred, label):
        pred = tf.reshape(pred,[-1])
        label = tf.reshape(label,[-1])
        return tf.reduce_mean(tf.square(pred-label))

    def constructingModel(self, inputTensor= None):
        self.placeholderInit()
        self.weights = weights = self.weightInit()
        
        def metaLearn_task(dictionaryTensor, reuse=True):
            input_1, input_2, label_1, label_2 = inputTensor
            output_task_1 = self.forwardStep(input_1, weights, reuse= reuse)
            loss_task_1 = self.lossfunction(output_task_1, label_1)
            grads = tf.gradients(loss_task_1, list(weights.values()))
            gradients = dict(zip(weights.keys(), grads))
            fast_weights = dict(zip(weights.keys(), [weights[key] - self.adamlr*gradients[key] for key in weights.keys()]))

            output_task_2 = self.forwardStep(input_2, weights, reuse= reuse)
            








