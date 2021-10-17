import sklearn
from sklearn.ensemble import RandomForestRegressor

import tensorflow as tf
import keras
from keras.optimizers import Adam
from keras.models import Sequential 
from keras.layers import Dense, Dropout  
from keras.losses import MAE, Loss

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import gpflow as gpf
from gpflow.kernels import RBF

# Random forest model used for all Datasets
def rf_model():
    estimator = RandomForestRegressor(n_estimators= 400,
    min_samples_split= 2,
    min_samples_leaf=1,
    max_features='sqrt',
    max_depth= None,
    bootstrap= False)
    return(estimator)

# Gaussian process model used for all Datasets    
def gp_model():
    k = gpf.kernels.RBF()
    model = gpf.models.GPR(kernel=k, mean_function=None)
    model.likelihood.variance.assign(0.01)
    model.kernel.lengthscales.assign(0.3)
    opt = gpf.optimizers.Scipy()

    opt.minimize(model.training_loss, model.trainable_variables, options=dict(maxiter=100)) 
    return(model)

# Model for Dataset 1, with Dropout
def create_mlp_d1_dropout():  
    r'''Creates the MLP model for dataset 1 with 2 dropout (0.3) layers. 
        hidden_layers = [1300, 650, 325, 162, 1], 
        activation function = 'tanh'.
    '''
    
    model = Sequential() 
    model.add(Dense(1300, kernel_initializer = 'normal', activation = 'tanh',
    input_shape = (1300,)))
    model.add(Dropout(0.3))
    model.add(Dense(650, activation = 'tanh')) 
    model.add(Dropout(0.3))
    model.add(Dense(325, activation = 'tanh')) 
    model.add(Dense(162, activation = 'tanh'))
    model.add(Dense(1))
    model.compile(
        loss = 'mse', 
        optimizer = Adam(lr=10**(float(-2.36794715)),epsilon=1e-8), 
        metrics = ['mean_absolute_error']
        )
    return(model)

# Model for Dataset 1, no Dropout
def create_mlp_d1():  
    r'''Creates the MLP model for Dataset 1 without dropout. 
        hidden_layers = [1300, 650, 325, 162, 1], 
        activation function = 'tanh'.
    '''

    model = Sequential() 
    model.add(Dense(1300, kernel_initializer = 'normal', activation = 'relu',
                    input_shape = (1300,))) 
    model.add(Dense(650, activation = 'relu')) 
    model.add(Dense(325, activation = 'relu')) 
    model.add(Dense(162, activation = 'relu'))
    model.add(Dense(1))
    model.compile(
        loss = 'mse', 
        optimizer = Adam(lr=10**(float(-2)),epsilon=1e-8), 
        metrics = ['mean_absolute_error']
        )
    return(model)

# Model for Dataset SOAP, with Dropout
def create_mlp_soap():
        r'''Creates the MLP model the SOAP encoded dataset with dropout.'''
    model = Sequential()
    model.add(Dense(8232, kernel_initializer = 'normal', activation = 'tanh',
                input_shape = (8232,))),
    model.add(Dense(4704, activation = 'tanh')),
    model.add(Dense(3000, activation = 'tanh')),
    model.add(Dropout(0.1)),
    model.add(Dense(2000, activation = 'tanh')),
    model.add(Dense(960, activation = 'tanh')),
    model.add(Dropout(0.3)),
    model.add(Dense(760, activation = 'tanh')),
    model.add(Dropout(0.3)),
    model.add(Dense(510, activation = 'tanh')),
    model.add(Dense(60, activation = 'tanh')),
    model.add(Dense(1))  

    model.compile(
        loss = 'mse', 
        optimizer = Adam(lr=0.0001,epsilon=1e-8), 
        metrics = ['mean_absolute_error']
        )
    return(model)

# Scaled MAE Loss
class MAE_scaled(keras.losses.Loss):
    def __init__(self, e_max):
        super().__init__()
        self.e_max = e_max

    def call(self, y_true, y_pred):
        return keras.losses.MAE(y_true * self.e_max, y_pred * self.e_max)

# Model for Dataset 2, with Dropout
def create_mlp_d2():
    r'''Creates the MLP model for Dataset 2 with dropout.'''
    dropout = 0
    act = 'relu'

    input_shape = (1324,)
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    x = tf.keras.layers.Dense(5000, kernel_initializer = 'normal', activation = act)(x)
    x = tf.keras.layers.Dense(2670, activation = act)(x)
    x = tf.keras.layers.Dense(10, activation = act)(x)
    x = tf.keras.layers.Dense(5000, activation = act)(x)
    x = tf.keras.layers.Dense(5000, activation = act)(x)
    x = tf.keras.layers.Dense(10, activation = act)(x)
    x = tf.keras.layers.Dense(5000, activation = act)(x)
    x = tf.keras.layers.Dense(5000, activation = act)(x)
    x = tf.keras.layers.Dense(10, activation = act)(x)
    x = tf.keras.layers.Dense(5000, activation = act)(x)
    x = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    
    hp_learning_rate = 0.000001

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss='msle',
                metrics=[MAE_scaled(max_e)])
    return model

# Model for Dataset 3, with Dropout
def create_mlp_d3(): 
r'''Creates the MLP model for Dataset 3 with dropout.'''
    dropout = 0.007
    act = 'relu'

    input_shape = (2624,)
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    x = tf.keras.layers.Dense(3824, kernel_initializer = 'normal', activation = act)(x)
    x = tf.keras.layers.Dense(5000, activation = act)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(10, activation = act)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(10, activation = act)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(5000, activation = act)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(5000, activation = act)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(10, activation = act)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(10, activation = act)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(10, activation = act)(x)
    x = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    
    hp_learning_rate = 0.000001

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss='msle',
                metrics=[MAE_scaled(max_e)])
    return model

# Model for Dataset 4, with Dropout
def create_mlp_d4(): 
    r'''Creates the MLP model for Dataset 4 with dropout.'''
    dropout = 0.00
    act = 'relu'

    input_shape = (6024,)
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    x = tf.keras.layers.Dense(5000, kernel_initializer = 'normal', activation = act)(x)
    x = tf.keras.layers.Dense(10, activation = act)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(10, activation = act)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(5000, activation = act)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    
    hp_learning_rate = 0.000001

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss='msle',
                metrics=[MAE_scaled(max_e)])
    return model