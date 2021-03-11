import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import sys
import random
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

optimizer=sys.argv[1]
batch_size=int(sys.argv[2])
nepoch=int(sys.argv[3])
flag=sys.argv[4]
datalabel=sys.argv[5]
zdim=int(sys.argv[6])
choose=int(sys.argv[7])
activation=sys.argv[8]
datascaler=sys.argv[9]
x_path = sys.argv[10]
y_path = sys.argv[11]

scalers_dict_x = {"StandardScaler": StandardScaler(), "MinMaxScaler": MinMaxScaler(), "MaxAbsScaler": MaxAbsScaler(), "RobustScaler": RobustScaler(), "PowerTransformer(yj)": PowerTransformer(method='yeo-johnson'), "QuantileTransformer-normal": QuantileTransformer(output_distribution='normal', random_state=69), "QuantileTransformer-uniform": QuantileTransformer(output_distribution='uniform', random_state=69), "Normalizer": Normalizer()}
f_scaler = scalers_dict_x[datascaler]

outlabel=f'{optimizer}_{datalabel}_{zdim}_{choose}_{activation}_{datascaler}'
'''
PERFORMANCE EVAL FUNCTIONS
'''

# import the metrics package from sci-kit learn
from sklearn import metrics
# we need a function to find the nearest value in a list
def find_nearest(array, value):
    array=np.asarray(array)
    idx = (np.abs(array-value)).argmin()
    return array[idx]
# we need a function which will compute and return the AUC and inverse-mistag at 0.5 efficiency
# be careful, this is a prototyping function only as it is agnostic to the direction of the cuts
# always check the cuts are consistently on one side for a useful classifier
def get_perf_stats(labels, measures):
    global activation
    global optimizer
    auc = metrics.roc_auc_score(labels, measures)
    fpr, tpr, thresholds = metrics.roc_curve(labels,measures)
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


x_train = np.load(x_path)
y_train = np.load(y_path)


original_dim = len(x_train[0])
print(original_dim)
x_train = x_train.astype('float32')
# rescale data
x_train = f_scaler.fit_transform(x_train)

# define the variational autoencoder properties
hidden_dim = 64  # use hidden layer with 64 nodes
latent_dim = zdim  # use a 1 dimensional latent space

inputs = keras.Input(shape=(original_dim,))
h1 = keras.layers.Dense(hidden_dim, activation=activation)(inputs)
h2 = keras.layers.Dense(hidden_dim, activation=activation)(h1)
h3 = keras.layers.Dense(hidden_dim, activation=activation)(h2)
z_mean = keras.layers.Dense(latent_dim)(h3)
z_log_var = keras.layers.Dense(latent_dim)(h3)

@tf.function
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.0, stddev=1.0)
    return z_mean + K.exp(0.5*z_log_var)*epsilon

# z is sampled from Gaussian(z_mean, z_log_var)
z = keras.layers.Lambda(sampling)([z_mean, z_log_var])

# create encoder
encoder = keras.Model(inputs, [z_mean, z_log_var, z], name='encoder')

# create decoder
latent_inputs = keras.Input(shape=(latent_dim, ), name='z_sampling')
x1 = keras.layers.Dense(hidden_dim, activation=activation)(latent_inputs)
x2 = keras.layers.Dense(hidden_dim, activation=activation)(x1)
x3 = keras.layers.Dense(hidden_dim, activation=activation)(x2)
outputs = keras.layers.Dense(original_dim)(x3)  # no activation
decoder = keras.Model(latent_inputs, outputs, name='decoder')

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = keras.Model(inputs, outputs, name='vae')

# custom loss function: reconstruction loss + KL divergence regularization
rec_loss = keras.losses.mean_squared_error(inputs, outputs)
rec_loss *= 5000

kl_loss = -0.5*K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(rec_loss + kl_loss)

# add losses and compile
vae.add_loss(vae_loss)
vae.compile(optimizer=optimizer)
print(vae.summary())

# custom callback
from tensorflow.keras.callbacks import Callback
auc_zmean2 = []
imtafe_zmean2 = []
auc_zmean2_shift = []
imtafe_zmean2_shift = []
wghts = []
encoder_wghts = []
auc_kl = []
imtafe_kl = []
medians = []
klhist = []
auc_zlogvar = []
imtafe_zlogvar = []
class MyEvaluation(Callback):
    def __init__(self, validation_data=(), interval=10):
        super(Callback, self).__init__()
        self.interval = interval
        self.X_val, self.y_val = validation_data
    
    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            encoded = encoder.predict(self.X_val, verbose=0)
            # get zmeans and z_logvars
            z_means = encoded[0]
            z_logvars = encoded[1]

            # calculate the mean of zmeans for each dimension, we get a vector of dim. of latent dim
            z_means_mean = np.median(z_means, axis=0)
            print(f"zmeans_median: {z_means_mean}")
            print(f"zlogvar_median: {np.median(z_logvars, axis=0)}")
            medians.append(z_means_mean)

            # calculate zmean^2 by taking the square of each component, then summing
            z_means2 = np.sum(np.power(z_means, 2), axis=1)

            # calculate the shifted zmean^2 by taking the difference between the value and the mean for each component, squaring, then summing
            z_means2_shifted = np.sum(np.power(z_means-z_means_mean, 2), axis=1)

            # calculate performances of each classifier
            auc2, imtafe2 = get_perf_stats(self.y_val, z_means2)
            auc_zmean2.append(auc2)
            imtafe_zmean2.append(imtafe2)
            print(f"\nAt epoch {epoch+1} the zmean2 AUC is {np.round(auc2, 2)} and imtafe is {np.round(imtafe2, 2)}.\n")

            auc2sh, imtafe2sh = get_perf_stats(self.y_val, z_means2_shifted)
            auc_zmean2_shift.append(auc2sh)
            imtafe_zmean2_shift.append(imtafe2sh)
            print(f"\nAt epoch {epoch+1} the zmean2 shifted AUC is {np.round(auc2sh, 2)} and imtafe is {np.round(imtafe2sh, 2)}.\n")
            if zdim==1:
                auclv, imtafelv = get_perf_stats(np.reshape(self.y_val,-1), np.reshape(np.ndarray.flatten(z_logvars),-1))
                auc_zlogvar.append(auclv)
                imtafe_zlogvar.append(imtafelv)
                print(f"\nAt epoch {epoch+1} the zlogvar AUC is {np.round(auclv, 2)} and imtafe is {np.round(imtafelv, 2)}.\n")

            
            kls = -0.5*K.sum(1 + z_logvars - K.square(z_means) - K.exp(z_logvars), axis=-1)
            kl=K.mean(kls).numpy()
            print(f"kl: {kl}")
            klhist.append(kl)
            auckl, imtafekl = get_perf_stats(np.reshape(self.y_val,-1), np.reshape(kls,-1))
            auc_kl.append(auckl)
            imtafe_kl.append(imtafekl)
            print(f"\nAt epoch {epoch+1} the kl AUC is {np.round(auckl, 2)} and imtafe is {np.round(imtafekl, 2)}.\n")
            print("Saving...")

            # np.save(f"{flag}_history_{optimizer}_{sb}.npy", (history.history)['loss'])
            wghts.append(vae.get_weights())
            encoder_wghts.append(encoder.get_weights())
            np.save(f"runs/{flag}_wghts_{outlabel}.npy", np.asarray(wghts))
            np.save(f"runs/{flag}_encwghts_{outlabel}.npy", np.asarray(encoder_wghts))
            if zdim==1:
                np.save(f"runs/{flag}_perf_{outlabel}.npy", np.asarray([auc_zmean2, imtafe_zmean2, auc_zmean2_shift, imtafe_zmean2_shift, auc_kl, imtafe_kl, auc_zlogvar, imtafe_zlogvar]))
            else:
                np.save(f"runs/{flag}_perf_{outlabel}.npy", np.asarray([auc_zmean2, imtafe_zmean2, auc_zmean2_shift, imtafe_zmean2_shift, auc_kl, imtafe_kl]))
            np.save(f"runs/{flag}_medians_{outlabel}.npy", np.asarray(medians))
            np.save(f"runs/{flag}_kls_{outlabel}.npy", np.asarray(klhist))
# train on data

myEval = MyEvaluation(validation_data=[x_train, y_train], interval=1)

history = vae.fit(x_train, x_train, epochs=nepoch, batch_size=batch_size, validation_data=None, callbacks=[myEval], verbose=2)
np.save(f"runs/{flag}_history_{outlabel}.npy", (history.history)['loss'])

x_train_encoded = encoder.predict(x_train)
np.save(f"runs/{flag}_zs_{outlabel}.npy", np.asarray(x_train_encoded))
print("All done")