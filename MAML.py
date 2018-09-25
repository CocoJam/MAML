import tensorflow as tf
import numpy as np
from tensorflow.python.platform import flags
Flags = flags.FLAGS
class MAML:
    def __init__(self, inputTensor= None ,inputShape = 1, outShape = 1 , testUpdateNum = 5, numClass=1):
        self.inputShape = inputShape
        self.outShape = outShape
        self.update_lr = Flags.update_lr
        self.adamlr = Flags.meta_lr
        self.hidden = [40,40]
        self.numClass = numClass
        self.testUpdateNum = testUpdateNum
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
        for i in range(1, len(self.hidden)):
            weights["w"+ str(i+1)] = tf.Variable(tf.truncated_normal([self.hidden[i-1], self.hidden[i]], stddev=0.01))
            weights["b"+ str(i+1)] = tf.Variable(tf.zeros([self.hidden[i]]))
        weights['w'+str(len(self.hidden)+1)] = tf.Variable(tf.truncated_normal([self.hidden[-1], self.outShape], stddev=0.01))
        weights['b'+str(len(self.hidden)+1)] = tf.Variable(tf.zeros([self.outShape]))
        return weights
    
    def normalization(self, tensors, activationFunc, reuse,scopeName):
        
        if(Flags.norm == "batch_norm"):
           return tf.contrib.layers.batch_norm(tensors, activation_fn = activationFunc, reuse= reuse, scope = scopeName)
        elif(Flags.norm == "layer_norm"):
            return tf.contrib.layers.layer_norm(tensors, activation_fn = activationFunc, reuse= reuse, scope = scopeName)
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
            hidden = self.normalization(hidden, tf.nn.relu, reuse= reuse, scopeName = str(i+1))
            
        return  tf.add(tf.matmul(hidden, weight["w"+str(len(self.hidden)+1)]), weight["b"+ str(len(self.hidden)+1)])
    
    def lossfunction(self, pred, label):
        pred = tf.reshape(pred,[-1])
        label = tf.reshape(label,[-1])
        return tf.reduce_mean(tf.square(pred-label))

    def constructingModel(self, inputTensor= None):
        self.placeholderInit()

        with tf.variable_scope('model', reuse=None) as training_scope:
            self.weights = weights = self.weightInit()
            number_of_update = max(Flags.num_updates, self.testUpdateNum )

            def metaLearn_task(dictionaryTensor, reuse=True):
                input_1, input_2, label_1, label_2 = dictionaryTensor
            
                output_task_2_list, loss_task_2_list = [],[]
                output_task_1 = self.forwardStep(input_1, weights, reuse= reuse)
                
                loss_task_1 = self.lossfunction(output_task_1, label_1)
                # print(loss_task_1)
                grads = tf.gradients(loss_task_1, list(weights.values()))
                gradients = dict(zip(weights.keys(), grads))
                fast_weights = dict(zip(weights.keys(), [weights[key] - self.adamlr*gradients[key] for key in weights.keys()]))
                # print(fast_weights)
                output_task_2 = self.forwardStep(input_2, fast_weights, reuse= reuse)
                
                output_task_2_list.append(output_task_2)
                loss_task_2_list.append(self.lossfunction(output_task_2, label_2))

                for i in range (number_of_update -1):
                    output_task_1_2 = self.forwardStep(input_1, fast_weights, reuse= reuse)
                    loss_task_1_2 = self.lossfunction(output_task_1_2, label_1)
                    grads = tf.gradients(loss_task_1_2, list(fast_weights.values()))
                    gradients = dict(zip(fast_weights.keys(), grads))
                    fast_weights = dict(zip(fast_weights.keys(), [fast_weights[key] - self.update_lr*gradients[key] for key in fast_weights.keys()]))
                    output_task_2 = self.forwardStep(input_2, fast_weights, reuse=reuse)
                    output_task_2_list.append(output_task_2)
                    loss_task_2_list.append(self.lossfunction(output_task_2, label_2))

                task_output = [output_task_1, output_task_2_list, loss_task_1, loss_task_2_list]
                return task_output

            output_dtype = [tf.float32,[tf.float32] * number_of_update, tf.float32, [tf.float32] * number_of_update]

            output_task_1,output_task_2_list, loss_task_1, loss_task_2_list = tf.map_fn(metaLearn_task, elems=(self.input_1, self.input_2, self.label_1, self.label_2), dtype= output_dtype, parallel_iterations= Flags.meta_batch_size)
        self.pre_meta_loss = tf.reduce_mean(loss_task_1)/tf.to_float(Flags.meta_batch_size)
        # total_losses2 = [tf.reduce_sum(loss_task_2_list[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(number_of_update)]

        self.post_meta_loss = [ tf.reduce_mean(task_2_loss)/tf.to_float(Flags.meta_batch_size) for task_2_loss in loss_task_2_list]
        
        optimizer = tf.train.AdamOptimizer(Flags.meta_lr)
        gvs = optimizer.compute_gradients( self.post_meta_loss[number_of_update-1])
        self.metatrain_op = optimizer.apply_gradients(gvs)
        tf.summary.scalar('Pre-update loss', self.pre_meta_loss )
        for i in range(number_of_update):
            tf.summary.scalar('Post-update loss '+ str(i+1), self.post_meta_loss[i])

        

            








