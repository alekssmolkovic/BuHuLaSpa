'''
DESCRIPTION
'''

# the description below will be added to the config file in the experiments directory
description = "VAE description"

'''
STANDARD IMPORTS
'''

import os
import sys
import numpy as np
import time

'''
PARSING ARGUMENTS
'''

# argparse allows us to parse our arguments easily, and incorporate '--help' functionality
import argparse
# initalise the parser
parser = argparse.ArgumentParser()
# define the arguments we can take, both required positional arguments and optional arguments specified with the hyphens
parser.add_argument("expt", type=str, help="name of the experiment you want to run")
parser.add_argument("data", type=str, help="name of your .npy data file")
parser.add_argument("masses", type=str, help="name of your .npy file with invariant masses of events")
parser.add_argument("labels", type=str, help="name of your .npy labels file")
parser.add_argument("-d", "--dimensions", type=int, default=1, help="set number of latent dimensions, default = 1")
parser.add_argument("-e", "--epochs", type=int, default=50, help="set number of epochs to run for, default = 50")
parser.add_argument("-b", "--batchsize", type=int, default=128, help="set batch size to run with, default = 128")
parser.add_argument("-l", "--learningrate", type=float, default=0.001, help="set learning rate to use with the optimizer, default = 0.001")
parser.add_argument("-o", "--optimizer", type=str, default="adadelta", help="set the optimizer, default = adadelta",
        choices = ["adadelta", "adam", "adagrad", "nadam"])
parser.add_argument("-a", "--activation", type=str, default="selu", help="set the activation functions for the hidden layers, default = selu",
        choices = ["linear", "relu", "selu", "sigmoid", "tanh"])
parser.add_argument("-c", "--reconstructionloss", type=str, default="mse", help="set the form of the reconstruction error, default = mse",
        choices = ["mse", "mae", "msle", "mape", "kld", "huber"])
parser.add_argument("-i", "--architecture", type=str, default="double100", help="choose from pre-defined architectures, default = double100",
        choices = ["quadruple100","triple100","double100","single100","largecascade", "smallcascade", "triple64"])
parser.add_argument("-s", "--samples", type=int, default=1, help="number of times to sample per event for the re-parameterisation trick, default = 1")
parser.add_argument("-r", "--multireco", type=float, default=5000.0, help="multipler for the reconstruction loss term, default = 5000")
parser.add_argument("-k", "--multikl", type=float, default=1.0, help="multipler for the KL loss term, default = 1")
parser.add_argument("-w", "--weights", type=str, help="insert path to load and use pre-existing weights instead of training new ones")
parser.add_argument("-n", "--batchnorm", type=int, help="use batchnorm 0 or 1")

# then parse the arguments from the command line
args = parser.parse_args()

'''
DIRECTORY SET-UP
'''

# each experiment will go to it's own directory in ../experiments/
expt_tag=args.expt
expt_dir="experiments/"+expt_tag+"/"
# check if the experiment already exists there, and quit the job to prevent over-writing the data
if os.path.isdir(expt_dir):
    # sys.exit("ERROR: experiment already exists, don't want to overwrite it by mistake")
    print(f"ERROR: experiment {expt_tag} already exists, don't want to overwrite it by mistake")
    while(True):
        x=input("Do you want to continue overwriting (1) or exit (0): ")
        if x == "1":
            break
        elif x == "0":
            sys.exit("Quit the program")
        else:
            continue
# if it does not already exist, create it
else:
    os.makedirs(expt_dir)
# print the name of our experiment
print("experiment: "+str(args.expt))

'''
TENSORFLOW IMPORTS
'''

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import initializers
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, RobustScaler
# this sets the default num type for all layers used throughout
tf.keras.backend.set_floatx('float32')

# fix as tf tries to allocate too much memory on the GPU by default (if using GPUs)
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


'''
DATA LOADING FUNCTION
'''

# here we define a function for loading and pre-processing the data for input to the VAE, the only argument is an int to define the batch size
def load_data(batch_size):
    # initialise the scaler transforms for the inputs and the invariant masses
    maxabsScaler_x = MaxAbsScaler()
    maxabsScaler_m = MaxAbsScaler()
    # load the inputs, truth labels, and invariant masses
    x_train=np.load(args.data)
    y_train=np.load(args.labels)
    invmasses=np.load(args.masses)
    # determine the number of events
    num_events = len(x_train)
    # determine the dimension of the inputs
    original_dim = len(x_train[0])
    # define input and invariant masses as type 'float32'
    x_train = x_train.astype('float32')
    invmasses = invmasses.astype('float32')
    # fit the transformation for the inputs according to maxabs, and apply it to x_train
    x_train = maxabsScaler_x.fit_transform(x_train)
    # extract signal-only events so we can study these alone
    x_signal = np.asarray( [ x_train[i] for i in range(num_events) if y_train[i]==1 ] )
    x_signal = x_signal.astype('float32')
    # re-shape the invariant masses from a 1D vector to an array, so that the maxabs transform can be performed, then perform the transformation as for x_train
    invmasses_rescaled = maxabsScaler_m.fit_transform(invmasses.reshape(-1,1))
    # re-shape the invariant masses back to a 1D vector
    invmasses_rescaled = invmasses_rescaled.reshape(-1)
    invmasses_signal_rescaled = np.asarray( [ invmasses_rescaled[i] for i in range(num_events) if y_train[i]==1 ] )
    # now we use tensorflows Dataset module to slice the input data and the invariant mass data into batches for training
    train_dataset_filtered = (tf.data.Dataset.from_tensor_slices(x_train).batch(batch_size))
    invmass_dataset_rescaled_filtered = (tf.data.Dataset.from_tensor_slices(invmasses_rescaled).batch(batch_size))

    # now we return all of the information necessary to plug the data into the VAE
    return num_events, train_dataset_filtered, original_dim, x_train, y_train, x_signal, invmass_dataset_rescaled_filtered, invmasses_rescaled, invmasses, invmasses_signal_rescaled


'''
ENCODER DEFINITION
'''

# the encoder is defined from a class which inherits from the tf.keras.layers.Layer class
class Encoder(tf.keras.layers.Layer):
    # the init method is run when the encoder is initialised
    # takes the latent dimension as an int and the hidden dimensions as a list of ints specifying their size
    def __init__(self, latent_dim, hidden_dims):
        super(Encoder, self).__init__()
        self.use_batchnorm = args.batchnorm
        # initialise layers in a loop over hidden_dims
        # input layer does not need to be initialised
        self.dense_hidden_layers = [
                                    Dense(
                                        hid_dim,
                                        # set activation from argparse
                                        activation = args.activation,
                                        # we can set an initializer if we like
                                        #kernel_initializer = tf.keras.initializers.RandomUniform(minval=0.01, maxval=0.1, seed=None),
                                        #bias_initializer = tf.keras.initializers.RandomUniform(minval=0.01, maxval=0.1, seed=None)
                                        )
                                    for hid_dim in hidden_dims
                                    ]
        if self.use_batchnorm:
            self.batch_norm_layers = [tf.keras.layers.BatchNormalization()
                                      for hid_dim in hidden_dims]

        # initialise two outputs layers for the z_means and z_log_vars
        self.dense_mean_layer = Dense(latent_dim)
        self.dense_log_var_layer = Dense(latent_dim)

    def sampling(self, inputs, is_z=True):
        z_mean, z_log_var = inputs
        if is_z:
            # normal z sampling using the reparameterization trick
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5*z_log_var)*epsilon
        else:
            # sample invariant masses with a 10% error
            epsilon = tf.keras.backend.random_normal(shape=z_mean.shape)
            ret = z_mean + tf.exp(0.5*z_log_var)*epsilon
            return tf.reshape(ret, (ret.shape[0], 1))

    # the call method is executed when we call the encoder, this is what constructs the encoder by glueing the layers together
    def call(self, inputs):
        # hidden layers will be saved in the list hs
        normal_inputs = inputs[0]
        m_inv_inputs = inputs[1]
        hs = []
        # we loop through the hidden layers that were initialised in the init method
        for i, hid_lay in enumerate(self.dense_hidden_layers):
            if i==0:
                hs.append(hid_lay(normal_inputs))
            else:
                if self.use_batchnorm:  # can use batch normalization
                    hs.append(self.batch_norm_layers[i-1](hid_lay(hs[i-1]),
                                                          training=False))
                else:
                    hs.append(hid_lay(hs[i-1]))
        # the output from the hidden layers are saved in the list h
        if self.use_batchnorm:
            h = self.batch_norm_layers[-1](hs[-1], training=False)
        else:
            h = hs[-1]

        # the outputs from the hidden layer are then passed to the z_mean and z_log_var layers that were initialised in the init method
        z_mean = self.dense_mean_layer(h)
        z_log_var = self.dense_log_var_layer(h)
        # then from the outputs of these layers the sampling step is performed using the sampling method defined above
        # z = [ self.sampling((z_mean, z_log_var)) for i in range(args.samples) ]
        z = self.sampling((z_mean, z_log_var), is_z=True)
        # sample the invariant mass with a 10% error
        m_inv_mean = m_inv_inputs
        m_inv_log_var = tf.math.log((0.1*m_inv_mean)**2)
        m_inv_rnd = self.sampling((m_inv_mean, m_inv_log_var), is_z=False)
        # finally, when the encoder is called, it returns the means, variances, and sampled z's + invariant masses calculated for a single batch
        return z_mean, z_log_var, z, m_inv_rnd

'''
DECODER DEFINITION
'''

# the decoder is defined from a class which inherits from the tf.keras.layers.Layer class
class Decoder(tf.keras.layers.Layer):
    # the init method is run when the encoder is initialised
    # takes the dimension of the inputs data as an int and the hidden dimensions as a list of ints specifying their size
    # we don't specify the latent dimension now because this is the size of the input layer, which is automatically calculated from the data
    def __init__(self, original_dim, hidden_dims):
        super(Decoder, self).__init__()
        # initialise layers in a loop over hidden_dims
        self.use_batchnorm = args.batchnorm
        self.dense_hidden_layers = [
                                    Dense(
                                        hid_dim,
                                        # set activation from argparse
                                        activation = args.activation,
                                        # we can set an initializer if we like
                                        #kernel_initializer = tf.keras.initializers.RandomUniform(minval=0.01, maxval=0.1, seed=None),
                                        #bias_initializer = tf.keras.initializers.RandomUniform(minval=0.01, maxval=0.1, seed=None)
                                        )
                                    for hid_dim in hidden_dims[::-1]  # should flip the hidden layers this time, to be symmetric wrt the encoder

                                    ]
        if self.use_batchnorm:
            self.batch_norm_layers = [tf.keras.layers.BatchNormalization()
                                      for hid_dim in hidden_dims]
        # the output layer is then defined with the dimension of the inputs
        self.dense_output = Dense(original_dim)

    # the call method is executed when we call the decoder, this is what constructs the decoder by glueing the layers together
    def call(self, inputs):
        # hidden layers will be saved in the list hs
        hs = []
        # we loop through the hidden layers that were initialised in the init method
        for i, hid_lay in enumerate(self.dense_hidden_layers):
            if i==0:
                # hs.append(hid_lay(tf.concat([inputs,tf.transpose(imasses)],axis=1)))
                hs.append(hid_lay(inputs))
            else:
                if self.use_batchnorm:
                    hs.append(self.batch_norm_layers[i-1](hid_lay(hs[i-1]),
                                                          training=False))
                else:
                    hs.append(hid_lay(hs[i-1]))
        # the output from the hidden layers are saved in the list h
        if self.use_batchnorm:
            h = self.batch_norm_layers[-1](hs[-1], training=False)
        else:
            h = hs[-1]
        # the outputs are simply the numbers on the last layer of the VAE, which are returned by this call method
        return self.dense_output(h)

'''
VAE MODEL DEFINITION
'''

# The VAE itself is defined as a class which inherits from the tf.keras.Model class
# it constructs the VAE model from the encoder and decoder
class iVAE(tf.keras.Model):
    # in the init method, executed when the VAE model is initiated, we must define the dimension of the inputs and the latent dimension as ints, and the hidden_dims as a list of ints
    def __init__(self, original_dim, hidden_dims, latent_dim):
        super(iVAE, self).__init__()
        # define the dimension of the data for this VAE
        self.original_dim = original_dim
        # intialise the encoder and decoder for this VAE
        self.encoder = Encoder(latent_dim=latent_dim, hidden_dims=hidden_dims)
        self.decoder = Decoder(original_dim, hidden_dims=hidden_dims)

    # the call method will be executed when the VAE is called, and as input it takes the input data from a single batch
    def call(self, inputs):
        # this batch is passed through the encoder by calling the encoder below, returning the means, variances, and sampled z's
        z_mean, z_log_var, z, m_inv_rnd = self.encoder(inputs)
        # we pass the sampled z's and invariant masses for the batch to the decoder which returns the ouput data for each element of the batch, which we save as reconstructed
        reconstructed = self.decoder(tf.concat([z, m_inv_rnd], 1))
        # finally we return reconstructed, i.e. the outputs from the decode for a single batch
        return z_mean, z_log_var, [reconstructed]

'''
LOSS FUNCTION SET-UP
'''
# here we define different loss functions that we could use for the reconstruction loss
mse_loss_fn = tf.keras.losses.MeanSquaredError()
mse_loss_fn_noreduce = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
mae_loss_fn = tf.keras.losses.MeanAbsoluteError()
mae_loss_fn_noreduce = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
msle_loss_fn = tf.keras.losses.MeanSquaredLogarithmicError()
msle_loss_fn_noreduce = tf.keras.losses.MeanSquaredLogarithmicError(reduction=tf.keras.losses.Reduction.NONE)
mape_loss_fn = tf.keras.losses.MeanAbsolutePercentageError()
mape_loss_fn_noreduce = tf.keras.losses.MeanAbsolutePercentageError(reduction=tf.keras.losses.Reduction.NONE)
kld_loss_fn = tf.keras.losses.KLDivergence()
kld_loss_fn_noreduce = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)
huber_loss_fn = tf.keras.losses.Huber(delta=0.01)
huber_loss_fn_noreduce = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE,delta=0.01)
# we define a multiplier which rescales the reconstruction loss and or kl loss
multireco = args.multireco
multikl = args.multikl
# below we define the function to compute the reconstruction loss
def reco_loss_fn(inputs,reconstructed):
    if args.reconstructionloss == "mse":
        return multireco * tf.reduce_mean( [ mse_loss_fn(inputs,ri) for ri in reconstructed ] )
    elif args.reconstructionloss == "mae":
        return multireco * tf.reduce_mean( [ mae_loss_fn(inputs,ri) for ri in reconstructed ] )
    elif args.reconstructionloss == "msle":
        return multireco * tf.reduce_mean( [ msle_loss_fn(inputs,ri) for ri in reconstructed ] )
    elif args.reconstructionloss == "mape":
        return multireco * tf.reduce_mean( [ mape_loss_fn(inputs,ri) for ri in reconstructed ] )
    elif args.reconstructionloss == "kld":
        return multireco * tf.reduce_mean( [ kld_loss_fn(inputs,ri) for ri in reconstructed ] )
    elif args.reconstructionloss == "huber":
        return multireco * tf.reduce_mean( [ huber_loss_fn(inputs,ri) for ri in reconstructed ] )
# below we define the function to compute the reconstruction losses for individual events to be used for classification
def reco_losses_fn(inputs,reconstructed):
    if args.reconstructionloss == "mse":
        return tf.reduce_mean( [ mse_loss_fn_noreduce(inputs,ri) for ri in reconstructed ] , 0)
    if args.reconstructionloss == "mae":
        return tf.reduce_mean( [ mae_loss_fn_noreduce(inputs,ri) for ri in reconstructed ] , 0)
    if args.reconstructionloss == "msle":
        return tf.reduce_mean( [ msle_loss_fn_noreduce(inputs,ri) for ri in reconstructed ] , 0)
    if args.reconstructionloss == "mape":
        return tf.reduce_mean( [ mape_loss_fn_noreduce(inputs,ri) for ri in reconstructed ] , 0)
    if args.reconstructionloss == "kld":
        return tf.reduce_mean( [ kld_loss_fn_noreduce(inputs,ri) for ri in reconstructed ] , 0)
    if args.reconstructionloss == "huber":
        return tf.reduce_mean( [ huber_loss_fn_noreduce(inputs,ri) for ri in reconstructed ] , 0)
# below we define the function to compute the reconstruction losses for individual observables to be used for classification
def reco_losses_obs_fn(inputs,reconstructed):
    if args.reconstructionloss == "mse":
        return tf.reduce_mean( [ mse_loss_fn_noreduce(np.transpose(inputs),np.transpose(ri)) for ri in reconstructed ] , 0)
    if args.reconstructionloss == "mae":
        return tf.reduce_mean( [ mae_loss_fn_noreduce(np.transpose(inputs),np.transpose(ri)) for ri in reconstructed ] , 0)
    if args.reconstructionloss == "msle":
        return tf.reduce_mean( [ msle_loss_fn_noreduce(np.transpose(inputs),np.transpose(ri)) for ri in reconstructed ] , 0)
    if args.reconstructionloss == "mape":
        return tf.reduce_mean( [ mape_loss_fn_noreduce(np.transpose(inputs),np.transpose(ri)) for ri in reconstructed ] , 0)
    if args.reconstructionloss == "kld":
        return tf.reduce_mean( [ kld_loss_fn_noreduce(np.transpose(inputs),np.transpose(ri)) for ri in reconstructed ] , 0)
    if args.reconstructionloss == "huber":
        return tf.reduce_mean( [ huber_loss_fn_noreduce(np.transpose(inputs),np.transpose(ri)) for ri in reconstructed ] , 0)

# here we define a function to compute the KL loss for the batch
def kl_loss_fn(zm,zlv):
    return -multikl*0.5*tf.reduce_mean( zlv - tf.square(zm) - tf.exp(zlv) + 1 )
# below we define a function to compute the KL loss for individual events to be used for classification
def kl_losses_fn(zm,zlv):
    return -multikl*0.5*tf.reduce_mean( zlv - tf.square(zm) - tf.exp(zlv) + 1 , 1)

'''
PERFORMANCE EVAL FUNCTIONS
'''

# import the metrics package from sci-kit learn
from sklearn import metrics
# we need a function to find the nearest value in a list
def find_nearest(array,value):
    array=np.asarray(array)
    idx = (np.abs(array-value)).argmin()
    return array[idx]
# we need a function which will compute and return the AUC and inverse-mistag at 0.5 efficiency
def get_perf_stats(labels,measures):
    auc = metrics.roc_auc_score(labels,measures)
    fpr,tpr,thresholds = metrics.roc_curve(labels,measures)
    try:
        imtafe = 1/fpr[list(tpr).index(find_nearest(list(tpr),0.5))]
    except:
        imtafe = 1
    if auc<0.5:
        measures = [-i for i in measures]
        auc = metrics.roc_auc_score(labels,measures)
        fpr,tpr,thresholds = metrics.roc_curve(labels,measures)
        try:
            imtafe = 1/fpr[list(tpr).index(find_nearest(list(tpr),0.5))]
        except:
            imtafe = 1
    return auc, imtafe

'''
TRAINING SET-UP
'''

print(f'  ------  LOADING/PREPROCESSING DATA AND INITIALISING NETWORK  ------  ')
# record time
time_0 = time.time()
# here we choose which optimizer to use, Adam or AdaDelta, we found AdaDelta to be best
if args.optimizer == "adadelta":
    optimizer = tf.keras.optimizers.Adadelta(learning_rate=args.learningrate, rho=0.95, epsilon=1e-07)
elif args.optimizer == "adam":
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learningrate)
elif args.optimizer == "adagrad":
    optimizer = tf.keras.optimizers.Adagrad(learning_rate=args.learningrate)
elif args.optimizer == "nadam":
    optimizer = tf.keras.optimizers.Nadam(learning_rate=args.learningrate)
# batch size is loaded from the arg parser and the hidden dimensions and latent dimensions are defined
batch_size = args.batchsize
if args.architecture == "quadruple100":
    hidden_dims = [100,100,100,100]
elif args.architecture == "triple100":
    hidden_dims = [100,100,100]
elif args.architecture == "double100":
    hidden_dims = [100,100]
elif args.architecture == "single100":
    hidden_dims = [100]
elif args.architecture == "smallcascade":
    hidden_dims = [7,6,5,4,3,2]
elif args.architecture == "largecascade":
    hidden_dims = [64,56,48,40,32,24,16,8]
elif args.architecture == "triple64":
    hidden_dims = [64,64,64]
latent_dim = args.dimensions
# we load the data using the function defined at the beginning
num_events, train_dataset, original_dim, x_train, y_train, x_signal, invmass_dataset_rescaled, invmasses_rescaled, invmasses, invmasses_rescaled_signal = load_data(batch_size)

# IMPORTANT: for iVAE we need to also use the invariant mass
train_dataset = list(zip(train_dataset, invmass_dataset_rescaled))
x_train = [x_train, invmasses_rescaled]
x_signal = [x_signal, invmasses_rescaled_signal]

# we initialise the VAE using the iVAE model class defined above
vae = iVAE(original_dim, hidden_dims, latent_dim)
# record time
time_1 = time.time()
# print out time taken for epoch to complete
print("time taken to load and preprocess data: "+str(time_1-time_0)+" s")

'''
SAVING RECORD
'''

# here we save a record of the run in the experiment directory so that we know what parameters the experiment used etc
with open(expt_dir+expt_tag+"_config.txt","a") as cfg:
    cfg.write("description: "+description+"\n")
    cfg.write("script: "+str(sys.argv[0])+"\n")
    cfg.write("epochs: "+str(args.epochs)+"\n")
    cfg.write("batchsize: "+str(args.batchsize)+"\n")
    cfg.write("learningrate: "+str(args.learningrate)+"\n")
    cfg.write("optimizer: "+str(args.optimizer)+"\n")
    cfg.write("activation: "+str(args.activation)+"\n")
    cfg.write("architecture: "+str(args.architecture)+"\n")
    cfg.write("samples: "+str(args.samples)+"\n")
    cfg.write("multireco: "+str(multireco)+"\n")
    cfg.write("multikl: "+str(multikl)+"\n")
    cfg.write("data: "+str(args.data)+"\n")
    cfg.write("labels: "+str(args.labels)+"\n")
    cfg.write("batchnorm: "+str(args.batchnorm)+"\n")
    if args.weights:
        cfg.write("weights loaded from: "+str(args.weights)+"\n")

'''
TRAINING LOOP
'''

# initialise lists to store statistics over the epochs
epoch_losses = []
epoch_recolosses = []
epoch_recolossesobs = []
epoch_kllosses = []
epoch_siglosses = []
epoch_sigrecolosses = []
epoch_sigrecolossesobs = []
epoch_sigkllosses = []
epoch_zmaucs = []
epoch_zmimtafes = []
epoch_zvaucs = []
epoch_zvimtafes = []
epoch_recoaucs = []
epoch_recoimtafes = []
epoch_klaucs = []
epoch_klimtafes = []
epoch_zmeans = []
epoch_zmeans_std = []
epoch_zlogvars = []
epoch_sigzmeans = []
epoch_sigzmeans_std = []
epoch_sigzlogvars = []

# set number of epochs using arg parse
nEpochs=args.epochs

# if the weights have not been provided as an input, run the training loop
if not args.weights:
    weights_history = []
    # looping through epochs
    for epoch in range(nEpochs):

        if epoch>0:
            print(f'  ------  START EPOCH {epoch}  ------  ')
        # record time
        time_0 = time.time()
        # looping through batches
        for step, x_batch_train in enumerate(train_dataset):
            # now we open GradientTape to compute the losses and their gradients
            with tf.GradientTape() as tape:
                # now we call the VAE for this batch, which will compute the loss for the batch
                z_mean, z_log_var, reconstructed = vae(x_batch_train)
                # define the reco loss
                reco_loss = reco_loss_fn(x_batch_train[0], reconstructed)
                # the above term is the reconstruction loss, and below we define the KL loss
                kl_loss = kl_loss_fn(z_mean,z_log_var)
                # sum the losses
                loss = reco_loss + kl_loss
            # assign the gradients of the loss function to grads
            grads = tape.gradient(loss, vae.trainable_weights)
            # apply the gradients to the weights of the VAE using the optimizer
            optimizer.apply_gradients(zip(grads, vae.trainable_weights))

        # at the beginning of the training we want a summary of the VAE architecture
        if epoch==0:
            print(vae.summary())
            print(f'  ------  START EPOCH {epoch}  ------  ')

        # record time
        time_1 = time.time()

        # we want to compute some statistics at the end of each epoch using the most recent weights
        # so we re-compute:
        z_mean_e, z_log_var_e, reconstructed_e = vae(x_train)
        reco_loss_e = reco_loss_fn(x_train[0],reconstructed_e)/multireco
        reco_losses_obs_e = reco_losses_obs_fn(x_train[0],reconstructed_e)
        kl_loss_e = kl_loss_fn(z_mean_e,z_log_var_e)/multikl
        loss_e = reco_loss_e + kl_loss_e
        reco_losses_e = reco_losses_fn(x_train[0],reconstructed_e)
        kl_losses_e = kl_losses_fn(z_mean_e,z_log_var_e)/multikl
        # and for signal-only:
        z_mean_se, z_log_var_se, reconstructed_se = vae(x_signal)
        reco_loss_se = reco_loss_fn(x_signal[0],reconstructed_se)/multireco
        reco_losses_obs_se = reco_losses_obs_fn(x_signal[0],reconstructed_se)
        kl_loss_se = kl_loss_fn(z_mean_se,z_log_var_se)/multikl
        loss_se = reco_loss_se + kl_loss_se
        # print losses
        print(" -- losses --")
        print("total loss: "+str(loss_e.numpy()))
        print("reco loss: "+str(reco_loss_e.numpy()))
        print("reco loss (per obs): "+str(reco_losses_obs_e.numpy()))
        print("kl loss: "+str(kl_loss_e.numpy()))
        # print signal losses
        print(" -- signal losses --")
        print("total signal loss: "+str(loss_se.numpy()))
        print("signal reco loss: "+str(reco_loss_se.numpy()))
        print("signal reco loss (per obs): "+str(reco_losses_obs_se.numpy()))
        print("signal kl loss: "+str(kl_loss_se.numpy()))
        # append them to stats lists
        epoch_losses.append(loss_e)
        epoch_recolosses.append(reco_loss_e)
        epoch_recolossesobs.append(reco_losses_obs_e)
        epoch_kllosses.append(kl_loss_e)
        epoch_siglosses.append(loss_se)
        epoch_sigrecolosses.append(reco_loss_se)
        epoch_sigrecolossesobs.append(reco_losses_obs_se)
        epoch_sigkllosses.append(kl_loss_se)
        # get performance stats at each epoch, separately using the norm of zmeans, the reco loss, and the KL loss as the classifier
        zm_auc_e, zm_imtafe_e = get_perf_stats(np.reshape(y_train,-1),np.reshape(np.linalg.norm(z_mean_e,axis=1),-1))
        zv_auc_e, zv_imtafe_e = get_perf_stats(np.reshape(y_train,-1),np.reshape(tf.reduce_mean(z_log_var_e,1),-1))
        reco_auc_e, reco_imtafe_e = get_perf_stats(np.reshape(y_train,-1),np.reshape(reco_losses_e,-1))
        kl_auc_e, kl_imtafe_e = get_perf_stats(np.reshape(y_train,-1),np.reshape(kl_losses_e,-1))
        # append to stats lists
        epoch_zmaucs.append(zm_auc_e)
        epoch_zmimtafes.append(zm_imtafe_e)
        epoch_zvaucs.append(zv_auc_e)
        epoch_zvimtafes.append(zv_imtafe_e)
        epoch_recoaucs.append(reco_auc_e)
        epoch_recoimtafes.append(reco_imtafe_e)
        epoch_klaucs.append(kl_auc_e)
        epoch_klimtafes.append(kl_imtafe_e)
        # print perf stats
        print(" -- performance --")
        print("AUC (zmeans): "+str(zm_auc_e))
        print("imtafe (zmeans): "+str(zm_imtafe_e))
        print("AUC (zvars): "+str(zv_auc_e))
        print("imtafe (zvars): "+str(zv_imtafe_e))
        print("AUC (reco): "+str(reco_auc_e))
        print("imtafe (reco): "+str(reco_imtafe_e))
        print("AUC (kl): "+str(kl_auc_e))
        print("imtafe (kl): "+str(kl_imtafe_e))
        # append averages of (abs) z_mean and z_log_var info to stats lists
        z_mean_mean_e = tf.reduce_mean(z_mean_e,0).numpy()
        z_mean_std_e = tf.math.reduce_std(z_mean_e,0).numpy()
        z_log_var_mean_e = tf.reduce_mean(z_log_var_e,0).numpy()
        epoch_zmeans.append(z_mean_mean_e)
        epoch_zmeans_std.append(z_mean_std_e)
        epoch_zlogvars.append(z_log_var_mean_e)
        # do the same for signal events only
        z_mean_mean_se = tf.reduce_mean(z_mean_se,0).numpy()
        z_mean_std_se = tf.math.reduce_std(z_mean_se,0).numpy()
        z_log_var_mean_se = tf.reduce_mean(z_log_var_se,0).numpy()
        epoch_sigzmeans.append(z_mean_mean_se)
        epoch_sigzmeans_std.append(z_mean_std_se)
        epoch_sigzlogvars.append(z_log_var_mean_se)
        # print VAE stats
        print(" -- VAE stats --")
        print("zmeans average: "+str(z_mean_mean_e))
        print("zmeans std: "+str(z_mean_std_e))
        print("zlogvars average: "+str(z_log_var_mean_e))
        print(" -- VAE signal-only stats --")
        print("zmeans average: "+str(z_mean_mean_se))
        print("zmeans std: "+str(z_mean_std_se))
        print("zlogvars average: "+str(z_log_var_mean_se))
        # save out the data collected over the epochs in numpy files
        np.save(expt_dir+expt_tag+"_losses.npy",epoch_losses)
        np.save(expt_dir+expt_tag+"_recolosses.npy",epoch_recolosses)
        np.save(expt_dir+expt_tag+"_recolossesobs.npy",epoch_recolossesobs)
        np.save(expt_dir+expt_tag+"_kllosses.npy",epoch_kllosses)
        np.save(expt_dir+expt_tag+"_siglosses.npy",epoch_siglosses)
        np.save(expt_dir+expt_tag+"_sigrecolosses.npy",epoch_sigrecolosses)
        np.save(expt_dir+expt_tag+"_sigrecolossesobs.npy",epoch_sigrecolossesobs)
        np.save(expt_dir+expt_tag+"_sigkllosses.npy",epoch_sigkllosses)
        np.save(expt_dir+expt_tag+"_zmaucs.npy",epoch_zmaucs)
        np.save(expt_dir+expt_tag+"_zmimtafes.npy",epoch_zmimtafes)
        np.save(expt_dir+expt_tag+"_zvaucs.npy",epoch_zvaucs)
        np.save(expt_dir+expt_tag+"_zvimtafes.npy",epoch_zvimtafes)
        np.save(expt_dir+expt_tag+"_recoaucs.npy",epoch_recoaucs)
        np.save(expt_dir+expt_tag+"_recoimtafes.npy",epoch_recoimtafes)
        np.save(expt_dir+expt_tag+"_klaucs.npy",epoch_klaucs)
        np.save(expt_dir+expt_tag+"_klimtafes.npy",epoch_klimtafes)
        np.save(expt_dir+expt_tag+"_zmeans.npy",epoch_zmeans)
        np.save(expt_dir+expt_tag+"_zmeans_std.npy",epoch_zmeans_std)
        np.save(expt_dir+expt_tag+"_zlogvars.npy",epoch_zlogvars)
        np.save(expt_dir+expt_tag+"_sigzmeans.npy",epoch_sigzmeans)
        np.save(expt_dir+expt_tag+"_sigzmeans_std.npy",epoch_sigzmeans_std)
        np.save(expt_dir+expt_tag+"_sigzlogvars.npy",epoch_sigzlogvars)
        # record time
        time_2 = time.time()

        # print out time taken for epoch to complete
        print(" -- time stats --")
        print("time taken for entire epoch: "+str(time_2-time_0)+" s")
        print("time taken to update network: "+str(time_1-time_0)+" s")
        print("time taken to get stats: "+str(time_2-time_1)+" s")

        # saving weights
        weights_history.append(vae.get_weights())
        np.save(expt_dir+expt_tag+"_wghts.npy", np.asarray(weights_history))

'''
LOADING WEIGHTS INSTEAD (IF PROVIDED)
'''

# if weights are provided as an argument we should not train the model and instead just load the weights
if args.weights:
    # to do this we need to first initialise the VAE with the dodgy work-around shown below
    for step, x_batch_train in enumerate(train_dataset):
        if step==0:
            reconstructed = vae(x_batch_train)
    # then we just load the weights and set them using 'set_weights'
    wghts = np.load(args.weights, allow_pickle=True)
    vae.set_weights(wghts)

'''
Can be extended with analysis using the provided weights.
'''

'''
THE END
'''
