import math
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import sys
import os
import sklearn as sk
import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score 
import matplotlib.pyplot as plt
import numpy as np
import csv

os.environ["CUDA_VISIBLE_DEVICES"]="3"

# Enable Eager Execution
tf.enable_eager_execution()

#from svnh_loader import SvnhLoader
from fn_loader import FnLoader
from tfrecord_loader import TfrecordLoader

from pi_model import PiModel, pi_model_loss, pi_model_gradients, ramp_up_function, ramp_down_function


def main():
    open('output_summary.csv', 'w').close()
    # Constants variables
    NUM_TRAIN_SAMPLES = 72485
    NUM_TEST_SAMPLES = 26528

    # Editable variables
    num_labeled_samples = 5126
    num_validation_samples = 0
    batch_size = 25
    epochs = 200
    max_learning_rate = 0.003
    initial_beta1 = 0.9
    final_beta1 = 0.5
    checkpoint_directory = './checkpoints/PiModel'
    tensorboard_logs_directory = './logs/PiModel'

    # Assign it as tfe.variable since we will change it across epochs
    learning_rate = tfe.Variable(max_learning_rate)
    beta_1 = tfe.Variable(initial_beta1)
    outputArr = np.array([])
    # Download and Save Dataset in Tfrecords
    #loader = SvnhLoader('./data', NUM_TRAIN_SAMPLES,
    #                    num_validation_samples, num_labeled_samples)
    #loader.download_images_and_generate_tf_record()
    loader = FnLoader('./fn_data', NUM_TRAIN_SAMPLES, num_validation_samples, num_labeled_samples)
#    print ("hello")
    loader.download_images_and_generate_tf_record()
    #sys.exit()
    # Generate data loaders
    train_labeled_iterator, train_unlabeled_iterator, validation_iterator, test_iterator = loader.load_dataset(
        batch_size, epochs)
    #print (train_labeled_iterator)
    batches_per_epoch = int(num_labeled_samples/batch_size)
    batches_per_epoch_val = int(num_validation_samples / batch_size)
#    sys.exit()
    model = PiModel()
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate, beta1=beta_1, beta2=0.999)
    max_unsupervised_weight = 100 * num_labeled_samples / \
        (NUM_TRAIN_SAMPLES - num_validation_samples)
    best_val_accuracy = 0
    global_step = tf.train.get_or_create_global_step()
    writer = tf.contrib.summary.create_file_writer(tensorboard_logs_directory)
    writer.set_as_default()
    #sys.exit()
    for epoch in range(epochs):

        rampdown_value = ramp_down_function(epoch, epochs)
        rampup_value = ramp_up_function(epoch)

        if epoch == 0:
            unsupervised_weight = 0
        else:
            unsupervised_weight = max_unsupervised_weight * \
                rampup_value

        learning_rate.assign(rampup_value * rampdown_value * max_learning_rate)
        beta_1.assign(rampdown_value * initial_beta1 +
                      (1.0 - rampdown_value) * final_beta1)
        epoch_loss_avg = tfe.metrics.Mean()
        epoch_accuracy = tfe.metrics.Accuracy()
        epoch_loss_avg_val = tfe.metrics.Mean()
        epoch_accuracy_val = tfe.metrics.Accuracy()
        for batch_nr in range(batches_per_epoch):
            X_labeled_train, y_labeled_train = train_labeled_iterator.get_next()
            #print(y_labeled_train[0:20,0])
            #print(y_labeled_train[0:20,1])
            #print(y_labeled_train.shape)
            X_unlabeled_train, _ = train_unlabeled_iterator.get_next()
       
            loss_val, grads = pi_model_gradients(X_labeled_train, y_labeled_train, X_unlabeled_train,
                                                 model, unsupervised_weight)
            optimizer.apply_gradients(zip(grads, model.variables),
                                      global_step=global_step)
            #sys.exit()
            epoch_loss_avg(loss_val)
            #print(X_labeled_train)
            num_test_batches = int(NUM_TEST_SAMPLES/batch_size)
            pred = model(X_labeled_train)
            #sys.exit()
            outputArr = np.array([])
            epoch_accuracy(tf.argmax(pred, 1), tf.argmax(y_labeled_train, 1))
            if (batch_nr == batches_per_epoch - 1):
                for test_batch in range(num_test_batches):
                    X_val, y_val = test_iterator.get_next()
                    y_val_predictions = model(X_val, training=False)
                    y_pred = tf.argmax(y_val_predictions, 1)
                    y_true = tf.argmax(y_val, 1)
                    y_pred_epoch = np.asarray(y_pred)
                    y_true_epoch = np.asarray(y_true)
                    #print(y_pred, y_true)
                    prec_epch = sk.metrics.precision_score(y_true_epoch, y_pred_epoch)
                    rec_epch = sk.metrics.recall_score(y_true_epoch, y_pred_epoch)
                    f1_epch = sk.metrics.f1_score(y_true_epoch, y_pred_epoch)

          
                    epoch_loss_avg_val(tf.losses.softmax_cross_entropy(
                        y_val, y_val_predictions))
                    epoch_accuracy_val(
                        tf.argmax(y_val_predictions, 1), tf.argmax(y_val, 1))
         #value1 = epoch+1
         #value2 = epoch_accuracy.result()
         #value3 = 
         #value4 = 
         #value5 =
         #value6 =
         #arrResult = [epoch+1, epoch_accuracy.result(), epoch_accuracy_val, a, b, c ]         
        arrResult = "{:03d}, {:02.6%}, {:02.6%}, {:.4%}, {:.4%}, {:.4%} ".format(epoch+1, epoch_accuracy.result(), epoch_accuracy_val.result(), prec_epch, rec_epch, f1_epch)
        out =  open('output_summary.csv', 'a+')
        out.write(arrResult + '\n')   
        #writef = csv.writer(out, delimiter=' ')
        #writef.writerow(arrResult)  
      
         

#        print("Epoch {:03d}/{:03d}: Train Loss: {:9.7f}, Train Accuracy: {:02.6%}, Validation Loss: {:9.7f}, "
#              "Validation Accuracy: {:02.6%}, lr={:.9f}, unsupervised weight={:5.3f}, beta1={:.9f}".format(epoch+1,
#                                                                                                           epochs,
#                                                                                                           epoch_loss_avg.result(),
#                                                                                                           epoch_accuracy.result(),
#                                                                                                           epoch_loss_avg_val.result(),
#                                                                                                           epoch_accuracy_val.result(),
#                                                                                                           learning_rate.numpy(),
#                                                                                                           unsupervised_weight,
#                                                                                                           beta_1.numpy()))
        print("Epoch {:03d}/{:03d}: Train Loss: {:9.7f}, Train Accuracy: {:02.6%}, lr={:.9f}, unsupervised weight={:5.3f}, beta1={:.9f}".format(epoch+1,
                                                                                                           epochs,
                                                                                                           epoch_loss_avg.result(),
                                                                                                           epoch_accuracy.result(),
                                                                                                           learning_rate.numpy(),
                                                                                                           unsupervised_weight,
                                                                                                           beta_1.numpy()))
        print (epoch_accuracy_val)
        #print (epoch_accuracy.result())
        # If the accuracy of validation improves save a checkpoint Best 85%
        if best_val_accuracy < epoch_accuracy.result():
            best_val_accuracy = epoch_accuracy.result()
            checkpoint = tfe.Checkpoint(optimizer=optimizer,
                                        model=model,
                                        optimizer_step=global_step)
            checkpoint.save(file_prefix=checkpoint_directory)

        # Record summaries
        #with tf.contrib.summary.record_summaries_every_n_global_steps(1):
        #    tf.contrib.summary.scalar('Train Loss', epoch_loss_avg.result())
        #    tf.contrib.summary.scalar(
        #        'Train Accuracy', epoch_accuracy.result())
        #    tf.contrib.summary.scalar(
        #        'Validation Loss', epoch_loss_avg_val.result())
        #    tf.contrib.summary.scalar(
        #        'Validation Accuracy', epoch_accuracy_val.result())
        #    tf.contrib.summary.scalar(
        #        'Unsupervised Weight', unsupervised_weight)
        #    tf.contrib.summary.scalar('Learning Rate', learning_rate.numpy())
        #    tf.contrib.summary.scalar('Ramp Up Function', rampup_value)
        #    tf.contrib.summary.scalar('Ramp Down Function', rampdown_value)
            

    #print('\nTrain Ended! Best Validation accuracy = {}\n'.format(best_val_accuracy))
    #sys.exit()
    # Load the best model
    root = tfe.Checkpoint(optimizer=optimizer,
                          model=model,
                          optimizer_step=tf.train.get_or_create_global_step())
    root.restore(tf.train.latest_checkpoint(checkpoint_directory))
  
    # Evaluate on the final test set
    #num_test_batches = NUM_TEST_SAMPLES/batch_size
    test_accuracy = tfe.metrics.Accuracy()
    #recall_eval = tf.metrics.recall(y_test_predictions, y_test)
    #precision_eval = tf.metrics.precision(y_test_predictions, y_test)
    for test_batch in range(int(num_test_batches)):
        X_test, y_test = test_iterator.get_next()
        #print(y_test[0:20,1])
        
        y_test_predictions = model(X_test, training=False)
        test_accuracy(tf.argmax(y_test_predictions, 1), tf.argmax(y_test, 1))
        y_pred = tf.argmax(y_test_predictions, 1)
        y_true = tf.argmax(y_test, 1)
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)
        #print(y_pred, y_true)
        a = sk.metrics.precision_score(y_true, y_pred)
        b = sk.metrics.recall_score(y_true, y_pred)
        c = sk.metrics.f1_score(y_true, y_pred)	

    print("Precision", a)
    print ("Recall", b)
    print ("f1_score", c)
    #print ("confusion_matrix")
    #print (sk.metrics.confusion_matrix(y_true, y_pred))
    #fpr, tpr, tresholds = sk.metrics.roc_curve(y_true, y_pred)




	#precision_eval = tf.metrics.precision(y_test_predictions, y_test)
    	#precision_eval = tf.contrib.metrics.precision_at_recall(tf.argmax(y_test_predictions, 1), tf.argmax(y_test, 1), 1) 
    print(tf.argmax(y_test_predictions))
    print(tf.argmax(y_test))
    #f1_score(y_test_predictions, y_test, average='macro')  
    print("Final Test Accuracy: {:.6%}".format(test_accuracy.result()))
   # print("Final Precision: {:.6%}".format(precision_eval.result()))
   # print("Final Recall: {:.6%}".format(recall_eval.result()))

if __name__ == "__main__":
    main()
