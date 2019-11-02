import os
import sys
import urllib

from scipy.io import loadmat
import numpy as np
import sys
import tensorflow as tf
import pandas as pd
from tfrecord_loader import TfrecordLoader


class FnLoader:
    

    # Constant attributes
    _NUM_TOTAL_SAMPLES = 99023
    #_TRAIN_URL = 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat'
    #_TEST_URL = 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat'
    _IMAGE_SIZE = [34,100, 1]
    _NUM_CLASSES = 2 # Fake and non-Fake

    def __init__(self, dataset_path, num_train_samples, num_validation_samples,
                 num_labeled_samples, random_seed=666):
        """ Init

        Arguments:
            dataset_path {string} -- the path to save the data
            num_train_samples {int} -- number of samples to use in training set (the sum of
                                       labeld + unlabeled train samples)
            num_validation_samples {int} -- number of samples to use in validation set
            num_labeled_samples {int} -- number of labeled samples to use
            random_seed {int} -- seed to use
        """

        self._dataset_path = dataset_path
        print (self._dataset_path)
        self._num_train_samples = num_train_samples
        print (self._num_train_samples)
        self._num_test_samples = self._NUM_TOTAL_SAMPLES - self._num_train_samples
        #self._num_test_samples = num_validation_samples
        print (self._num_test_samples)
        self._num_validation_samples = num_validation_samples
        print (self._num_validation_samples)
        self._num_labeled_samples = num_labeled_samples
        print (self._num_labeled_samples)
        self._num_unlabeled_train_samples = num_train_samples - \
            num_validation_samples - num_labeled_samples
        print (self._num_unlabeled_train_samples)
        self._random_seed = random_seed

    def __normalize_and_prepare_dataset(self, mat_dataset):
        """ Receives a mat dataset and normalized the data accordingly to the 
           described in the original paper (std normalization)

        Arguments:
            mat_dataset {dict} -- mat dict (scipy.io.loadmat) dataset directly loaded 
                                  from the url mat

        Returns:
            [np.ndarray] -- Images normalized and flattened (num_images x (32*32*3))
            [np.ndarray] -- Correspondent labels
        """

        # Convert data to numpy array
        X = mat_dataset['X'].astype(np.float64)

        # Convert it to zero mean and unit variance
        X -= np.mean(X, axis=(1, 2, 3), keepdims=True)
        X /= (np.mean(X ** 2, axis=(1, 2, 3), keepdims=True) ** 0.5)

        # Original dataset comes with wrong order in the dimensions
        X = X.transpose((3, 0, 1, 2))

        X = X.reshape([X.shape[0], -1])
        y = mat_dataset['y'].flatten().astype(np.int32)
        # 0 is label 10
        y[y == 2] = 0

        return X, y

    def __loadFn(self):
       
        fn_data_train  = pd.read_csv("./fn_data/trainData_syd.csv", delimiter=',', header=None).values
        fn_data_test = pd.read_csv("./fn_data/test_syd.csv", delimiter=',', header=None).values
        fn_data_unl = pd.read_csv("./fn_data/train_unlabeled_syd.csv", delimiter=',', header=None).values
        fn_data_train  = np.delete(fn_data_train, 0, axis=0)
        fn_data_test = np.delete(fn_data_test, 0, axis=0)
        fn_data_unl = np.delete(fn_data_unl, 0, axis=0)
        #print(fn_data_train[0, 5])
        #print(np.unique(fn_data_train[:, -1]))
        print(fn_data_train.shape, fn_data_test.shape, fn_data_unl.shape)
        temp = np.copy(fn_data_train[:, 0])
        fn_data_train[:, 0] = fn_data_train[:, -1]
        fn_data_train[:, -1] = temp
        #print(fn_data_train[:, -1])
        fn_data_train = np.concatenate((fn_data_train, fn_data_unl), axis=0)
        np.random.shuffle(fn_data_train)
        #print (fn_data_train.shape)
        #print(np.unique(fn_data_train[:, -1]))
	
        
 #       print ("hello")
        #sys.exit()
        return fn_data_train, fn_data_test, fn_data_unl

    def __extract_dataset(self):
        fn_data_train, fn_data_test, fn_data_unl = self.__loadFn()
        train_X = []
        train_y = []
        test_X = []
        test_y = []
        #      print (len(f_data[0]))
        #train_X, train_y = loadFn(filepath_train)
        #test_X, test_y = loadFn(filepath_test)
        #f_data = np.array(f_data)
        #nf_data = np.array(nf_data)
        #for row in fn_data_train:
        ##print(fn_data_train[:, 0])
        train_X, train_y = fn_data_train[:, : -1], fn_data_train[:, -1]
        print(train_X.shape)
        print(np.unique(train_y))
	
	#train_y = np.where(train_y==0, 2, train_y)
	        
        #for row in fn_data_test:
        test_X, test_y = fn_data_test[:, : -1], fn_data_test[:, -1]
        
        print(np.unique(test_y))
	#test_y = np.where(test_y==0, 1, test_y)
        print(test_y)
        
        #a = fn_data_unl[:, : -1]
	#b = fn_data_unl[:, -1]
	#print(a.shape, b.shape)
        #np.concantenate((train_X, a), axis=0)
        #np.concantenate((train_y, b), axis=None)
        print ('Datasets shape\n')
        print(train_X.shape, train_y.shape)
        print(test_X.shape, test_y.shape)
        
		#if i < 1997:
                #train_X.append(f_data[i])
                #train_y.append(1)
                #train_X.append(nf_data[i])
                #train_y.append(0)
            #else:
               #test_X.append(f_data[i])
                #test_y.append(1)
                #test_X.append(nf_data[i])
                #test_y.append(0)

        return np.array(train_X), np.array(train_y), np.array(test_X), np.array(test_y)
        
    def __download_and_extract_dataset(self):
        """ Downloads the dataset and saves it in the _dataset_path. 
            Data is saved as a .mat file (as given by the original
            dataset). The mat file is then loaded and std normalized.

        Returns:
            [np.array] -- normalized train images
            [np.array] -- train labels
            [np.array] -- normalized test images
            [np.array] -- test labels
        """

        filepath_train = self._dataset_path + '/train_32x32.mat'
        filepath_test = self._dataset_path + '/test_32x32.mat'

        def download_progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %.1f%%' % (
                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        # Download dataset
        urllib.request.urlretrieve(
            self._TRAIN_URL, filepath_train, download_progress)
        urllib.request.urlretrieve(
            self._TEST_URL, filepath_test, download_progress)

        print('\n')

        # Load resultant mat files
        train_data = loadmat(filepath_train)
        test_data = loadmat(filepath_test)

        # Normalize between 0 and 1
        train_X, train_y = self.__normalize_and_prepare_dataset(train_data)
        test_X, test_y = self.__normalize_and_prepare_dataset(test_data)

        # Remove mat files
        os.remove(filepath_train)
        os.remove(filepath_test)

        return train_X, train_y, test_X, test_y

    def __generate_tfrecord(self, images, labels, filename):
        """ Receives a set of images and labels and converts them into
            tensorflow tfrecords file saving them in the dataset path
            given with the desired filename.

        Arguments:
            images {np.array} -- images for this dataset
            labels {np.array} -- labels for this dataset
            filename {filename} -- filename for this dataset
        """

        # If we are taking care of unlabeled data
        if labels == []:
            pass
        elif images.shape[0] != labels.shape[0]:
            raise ValueError("Images size %d does not match label size %d." %
                             (images.shape[0], labels.shape[0]))

        print('Writing', filename)

        writer = tf.python_io.TFRecordWriter(filename)

        # Write each image for the tfrecords file
        for index in range(images.shape[0]):
            image = images[index].tolist()

            # If unlabeled dataset label is -1
            if labels == []:
                current_label = -1
            else:
                current_label = int(labels[index])

            sample = tf.train.Example(features=tf.train.Features(feature={
                'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[34])),
                'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[100])),
                'depth': tf.train.Feature(int64_list=tf.train.Int64List(value=[1])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[current_label])),
                'image': tf.train.Feature(float_list=tf.train.FloatList(value=image))}))
            writer.write(sample.SerializeToString())
        writer.close()

    def download_images_and_generate_tf_record(self):
        """ Main function of the class that allows generating and saving the tfrecords
            for labeled train, unlabeled train, validation and test datasets.
        """

        # Create folder if needed
        #if not os.path.exists(self._dataset_path):
        #    os.makedirs(self._dataset_path)
        #else:  # Dataset already loaded
        #    return
        print (self._dataset_path)
        # Download and process dataset
        #train_X, train_y, test_X, test_y = self.__download_and_extract_dataset() 
        train_X, train_y, test_X, test_y = self.__extract_dataset()
	#a, b, fn_data_unl = self.__loadFn()
	#a = []
	#b = []
	#unlabeled_train_X = fn_data_unl[:, : -1]
        print (train_X.shape)
        print (train_y.shape)
        	
        # Use the seed provided
        rng = np.random.RandomState(self._random_seed)
        
        # I know I could initalize to zeros to avoid the appends, but it's only
        # done once, so let me have it
        labeled_train_X = np.empty(shape=(0, 34*100*1))
        labeled_train_y = []
        unlabeled_train_X = np.empty(shape=(0, 34*100*1))
        validation_X = np.empty(shape=(0, 34*100*1))
        validation_y = []
        
        # Randomly shuffle the dataset, and have balanced labeled and validation
        # datasets (avoid having and unbalenced train set that could hurt the results)
        for label in range(2):
	             
            label_mask = (train_y == label)
            print(np.unique(label_mask))
            print (label_mask)
            current_label_X = train_X[label_mask]
            current_label_y = train_y[label_mask]
            current_label_X, current_label_y = rng.permutation(
                current_label_X), rng.permutation(current_label_y)
	    
            print (current_label_X.shape)
            print (current_label_y.shape)
            print (labeled_train_X.shape)
	    
            print (current_label_X[:int(self._num_labeled_samples/self._NUM_CLASSES), :].shape)
            # Take care of the labeled train set
            labeled_train_X = np.append(labeled_train_X, current_label_X[:int( self._num_labeled_samples/self._NUM_CLASSES), :], axis=0)
            labeled_train_y = np.append(labeled_train_y, current_label_y[:int( self._num_labeled_samples/self._NUM_CLASSES)])
            current_label_X = current_label_X[int(
                self._num_labeled_samples/self._NUM_CLASSES):, :]
            current_label_y = current_label_y[int(
                self._num_labeled_samples/self._NUM_CLASSES):]
            # Now let's take care of validation
            validation_X = np.append(validation_X, current_label_X[:int(
                self._num_validation_samples/self._NUM_CLASSES)], axis=0)
            validation_y = np.append(validation_y, current_label_y[:int(
                self._num_validation_samples/self._NUM_CLASSES)])
            current_label_X = current_label_X[int(
                self._num_validation_samples/self._NUM_CLASSES):, :]
            current_label_y = current_label_y[int(
                self._num_validation_samples/self._NUM_CLASSES):]
            # The rest goes to Unlabeled train
            #nlabeled_train_X = np.append(
               #unlabeled_train_X, current_labeled_X,  axis=0)

            print(unlabeled_train_X.shape)
        a, b, fn_data_unl = self.__loadFn()
        a = []
        b = []
        unlabeled_train_X = fn_data_unl[:, : -1]

        # Print final set shapes
        print("Labeled train shape: ", labeled_train_X.shape)
        print("Unlabeled train shape: ", unlabeled_train_X.shape)
        print("Validation shape: ", validation_X.shape)
        print("Test shape: ", test_X.shape)

        # Write tfrecords to disk
        self.__generate_tfrecord(labeled_train_X, labeled_train_y, os.path.join(
            self._dataset_path, 'labeled_train.tfrecords'))

        self.__generate_tfrecord(unlabeled_train_X, [], os.path.join(
            self._dataset_path, 'unlabeled_train.tfrecords'))

        self.__generate_tfrecord(validation_X, validation_y, os.path.join(
            self._dataset_path, 'validation_set.tfrecords'))

        self.__generate_tfrecord(test_X, test_y, os.path.join(
            self._dataset_path, 'test_set.tfrecords'))

    def load_dataset(self, batch_size, epochs):
        """ Calls the TfrecordLoader to load the generated 
           tfrecords file.

        Arguments:
            batch_size {int} -- desired batch size
            epochs {int} -- number of epochs for train

        Returns:
            {tf.data.Iterator} -- iterator for a specific tfrecords file
        """

        tfrecord_loader = TfrecordLoader(
            './fn_data', batch_size, epochs, self._IMAGE_SIZE, self._NUM_CLASSES)
        return tfrecord_loader.load_dataset()
