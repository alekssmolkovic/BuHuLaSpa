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


optimizer_str=sys.argv[1]
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
invm_path = sys.argv[12]
train_bool = int(sys.argv[13])
ii = sys.argv[14]
invm_relative_error = float(sys.argv[15])
zmean_zlogvar_kl_inputs = sys.argv[16] # a list with saved zmean, zlogvar, kl per event, equivalent to fixing the encoder - extract this at epoch of max kl

flag1=f'{flag}{zdim}{choose}{datalabel}'
# flag1=flag
flag_dec_only=f'iVAE{flag}deconly'

scalers_dict_x = {"StandardScaler": StandardScaler(), "MinMaxScaler": MinMaxScaler(), "MaxAbsScaler": MaxAbsScaler(), "RobustScaler": RobustScaler(), "PowerTransformer(yj)": PowerTransformer(method='yeo-johnson'), "QuantileTransformer-normal": QuantileTransformer(output_distribution='normal', random_state=69), "QuantileTransformer-uniform": QuantileTransformer(output_distribution='uniform', random_state=69), "Normalizer": Normalizer()}
f_scaler = scalers_dict_x[datascaler]

outlabel=f'{optimizer_str}_{datalabel}_{zdim}_{choose}_{datalabel}_{activation}_{datascaler}'

optimizer = 'adam'

x_train = np.load(x_path)
y_train = np.load(y_path)
invm = np.load(invm_path)

(zmean, zlogvar, kls) = np.load(zmean_zlogvar_kl_inputs)

original_dim = len(x_train[0])
print(original_dim)
x_train = x_train.astype('float32')


x_train = f_scaler.fit_transform(x_train)

invm = invm.astype('float32')
invm_mean = np.mean(invm)
invm_std = np.std(invm)

invm_rscld = (invm-np.mean(invm))/np.std(invm)
if invm_relative_error == 0.0:
    invm_rscld_logvars = 0.0*invm_rscld
else:
    invm_rscld_logvars = np.log(np.power((invm_relative_error*invm)/invm_std, 2))
invm_rscld = invm_rscld.reshape(-1, 1)
invm_rscld_logvars = invm_rscld_logvars.reshape(-1, 1)


# define the variational autoencoder properties
hidden_dim = 64  # use hidden layer with 64 nodes
latent_dim = zdim  # use a 1 dimensional latent space


@tf.function
def sampling(args):
    mean = args[0]
    logvar = args[1]
    epsilon = K.random_normal(shape=(K.shape(mean)[0], latent_dim), mean=0.0, stddev=1.0)
    if tf.equal(tf.reduce_sum(logvar), 0.0):
        return mean 
    else:
        return mean + K.exp(0.5*logvar)*epsilon

# create decoder
decoder_feature_fake_input = keras.Input(shape=(len(x_train[0]), ), name='fake_features_input')
zmean_input = keras.Input(shape=(1, ), name='zmean_input')
zlogvar_input = keras.Input(shape=(1, ), name='zlogvar_input')
invm_input = keras.Input(shape=(1, ), name='invm_input')
invm_logvar_input = keras.Input(shape=(1, ), name='invm_logvar_input')
z = keras.layers.Lambda(sampling)([zmean_input, zlogvar_input])
invm_smpld = keras.layers.Lambda(sampling)([invm_input, invm_logvar_input])
z_and_invm = keras.layers.Concatenate(axis=1)((z, invm_smpld))
x1 = keras.layers.Dense(hidden_dim, activation=activation)(z_and_invm)
x2 = keras.layers.Dense(hidden_dim, activation=activation)(x1)
x3 = keras.layers.Dense(hidden_dim, activation=activation)(x2)
outputs_dec = keras.layers.Dense(original_dim)(x3)  # no activation for our stuff
decoder = keras.Model(inputs=[decoder_feature_fake_input, zmean_input, zlogvar_input, invm_input, invm_logvar_input], outputs=outputs_dec, name='decoder')

rec_loss = K.mean(keras.losses.mean_squared_error(decoder_feature_fake_input, outputs_dec))
rec_loss *= 5000
decoder.add_loss(rec_loss)
# add losses and compile
decoder.compile(optimizer=optimizer)
print(decoder.summary())
# custom callback
from tensorflow.keras.callbacks import Callback
wghts = []
class MyEvaluation(Callback):
    def __init__(self, interval=10):
        super(Callback, self).__init__()
        self.interval = interval

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            wghts.append(decoder.get_weights())
            np.save(f"runs/{flag_dec_only}_wghts_{optimizer}_{datalabel}_{invm_relative_error}_{ii}.npy", np.asarray(wghts))
# train on data

myEval = MyEvaluation(interval=1)


if train_bool:
    if invm_relative_error!=0.0:
        print(f"including the invariant mass sampling step with relative error {invm_relative_error}")
        history = decoder.fit({"fake_features_input": x_train, "zmean_input": zmean, "zlogvar_input": zlogvar, "invm_input": invm_rscld, "invm_logvar_input": invm_rscld_logvars}, x_train, epochs=nepoch, batch_size=batch_size, validation_data=None, callbacks=[myEval], verbose=2)
    else:
        print("no invm sampling")
        history = decoder.fit({"fake_features_input": x_train, "zmean_input": zmean, "zlogvar_input": zlogvar, "invm_input": invm_rscld, "invm_logvar_input": 0.0*invm_rscld_logvars}, x_train, epochs=nepoch, batch_size=batch_size, validation_data=None, callbacks=[myEval], verbose=2)

    np.save(f"runs/{flag_dec_only}_history_{optimizer}_{datalabel}_{invm_relative_error}_{ii}.npy", (history.history)['loss'])

wghts = np.load(f"runs/{flag_dec_only}_wghts_{optimizer}_{datalabel}_{invm_relative_error}_{ii}.npy", allow_pickle=True)
decoder.set_weights(wghts[-1])
