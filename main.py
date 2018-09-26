import numpy as np
import random
import tensorflow as tf
from tensorflow.python.platform import flags
import pickle
import csv
from sin_wave_generation import sinGen
from MAML import MAML
Flags = flags.FLAGS

flags.DEFINE_string('datasource', 'sinusoid', 'sinusoid')
flags.DEFINE_integer('metatrain_iterations', 15000, 'number of metatraining iterations.')
flags.DEFINE_integer('meta_batch_size', 25, 'number of tasks sampled per meta-update')
flags.DEFINE_float('meta_lr', 0.001, 'the base learning rate of the generator')
flags.DEFINE_integer('update_batch_size', 10, 'number of examples used for inner gradient update (K for K-shot learning).')
flags.DEFINE_float('update_lr', 1e-3, 'step size alpha for inner gradient update.') 
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_bool('test', True, 'True to train, False to test.')
flags.DEFINE_integer('testNum', 1000, 'number of inner gradient update')
flags.DEFINE_integer('num_updates', 1, 'number of inner gradient update')
flags.DEFINE_string('norm', 'None', 'batch_norm, layer_norm, or None')
flags.DEFINE_string("summaries_dir", "tensorBroadLog", "TensorBroad location")
flags.DEFINE_string("logdir", "model", "Model location")

flags.DEFINE_bool("resume_model", True,"resume previous training model" )

Interval = 100
Big_interval = 1000
def main():
    if (Flags.datasource == "sinusoid"):
        dataGen = sinGen(Flags.update_batch_size*2, Flags.meta_batch_size)
       
        maml = MAML(inputTensor=None ,inputShape= dataGen.dim_input, outShape= dataGen.dim_output)
        maml.constructingModel(inputTensor=None)
        saver = loader = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=10)
        maml.summary_operation = tf.summary.merge_all()
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter(Flags.summaries_dir + '/train',sess.graph)
        iterNum =0

        if Flags.resume_model:
            model_file = tf.train.latest_checkpoint(Flags.logdir)
            print(model_file)
            if model_file:
                print("model resuming")
                ind1 = model_file.index('model')
                iterNum= int(model_file[ind1+11:])
                saver.restore(sess, model_file)

        if(Flags.train):
            train(iterNum, dataGen, maml, sess, train_writer, saver)
        if(Flags.test):
            test(dataGen, maml , Flags.testNum, sess)

def test(dataGen, maml, numTest, sess):
    num_classes = dataGen.classNum
    np.random.seed(1)
    random.seed(1)
    test_list = []
    for i in range(numTest):
        batch_x, batch_y, amp, phase = dataGen.generate()
        inputa = batch_x[:, :num_classes*Flags.update_batch_size, :]
        labela = batch_y[:, :num_classes*Flags.update_batch_size, :]
        inputb = batch_x[:, num_classes*Flags.update_batch_size:, :] # b used for testing
        labelb = batch_y[:, num_classes*Flags.update_batch_size:, :]
        feed_dict = {maml.input_1: inputa, maml.input_2: inputb,  maml.label_1: labela, maml.label_2: labelb}
        inputTensorOp = [maml.pre_meta_loss, maml.post_meta_loss[Flags.num_updates-1]]
        result = sess.run(inputTensorOp, feed_dict)
        test_list.append(result)
    accuracy = np.array(test_list)
    average = np.mean(accuracy, axis=0)
    standDev = np.std(accuracy, axis=0)
    ci95 = 1.96*standDev/np.sqrt(numTest)
    print("Average: " + str(average) + ", Std: "+ str(standDev)+ ", Confidence interval (95%): "+ str(ci95))
    resultDir = Flags.summaries_dir+"/test/" + "test_iteration_"+ str(numTest) + ".pkl"
    out_filename = Flags.summaries_dir+"/test/" + "test_iteration_"+ str(numTest) + "csv"
    with open(resultDir, 'wb') as f:
        pickle.dump({'test': accuracy}, f)
    with open(out_filename, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['update'+str(i) for i in range(len(average))])
        writer.writerow(average)
        writer.writerow(standDev)
        writer.writerow(ci95)




def train(iterNum, dataGen, maml ,sess, train_writer, saver):
    num_classes = dataGen.classNum
    preTrain_loss ,postTrain_loss = [],[]
    for i in range(iterNum,Flags.metatrain_iterations+1):
        batch_x, batch_y, amp, phase = dataGen.generate()
        inputa = batch_x[:, :num_classes*Flags.update_batch_size, :]
        labela = batch_y[:, :num_classes*Flags.update_batch_size, :]
        inputb = batch_x[:, num_classes*Flags.update_batch_size:, :] # b used for testing
        labelb = batch_y[:, num_classes*Flags.update_batch_size:, :]
        feed_dict = {maml.input_1: inputa, maml.input_2: inputb,  maml.label_1: labela, maml.label_2: labelb}
        inputTensorOp = [maml.metatrain_op]
        if(i % Interval == 0):
            inputTensorOp.extend([ maml.summary_operation ,maml.pre_meta_loss, maml.post_meta_loss[Flags.num_updates-1]])
        result = sess.run(inputTensorOp, feed_dict)
        if(i % Interval == 0):
            preTrain_loss.append(result[-2])
            postTrain_loss.append(result[-1])
            train_writer.add_summary(result[1],i)
        if(i% Big_interval == 0):
            print("Iteraction of "+ str(i)+ " :"+ str(np.mean(preTrain_loss)) + ", "+ str(np.mean(postTrain_loss)))
            saver.save(sess, Flags.logdir +  '/model' + str(i))
            preTrain_loss ,postTrain_loss = [],[]

main()