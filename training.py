import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
from kgcnn.utils.loss import ScaledMeanAbsoluteError
from utils import CosineLearningRateScheduler as lr
from input_generation import generate_input_data, painn_dataset
from models import make_painn

# Example workflow for the adapted PaiNN Network. 
# For other models just substitute the corresponding input generation and model generation functions

input_dir= "---"
output_dir="---"

#Generates input data from pdb and saves it in the given direction
generate_input_data(input_dir, output_dir)

#Loads the saved data, groups it to ragged tensors and splits into training, test and validation sets
split_size=0.2
xtrain,xtest,xval,ytrain,ytest,yval,scaler = painn_dataset(output_dir, split_size)

#Define the Loss, Metric and Optimizer parameters
learning_rate_start = 0.5e-3
learning_rate_stop = 1e-5
epo=10
epo_min=6
epostep = 10
cbks = lr(learning_rate_start, learning_rate_stop, epo_min, epo, verbose=0)
optimizer = tf.keras.optimizers.Adam(lr=learning_rate_start)
mae_metric = ScaledMeanAbsoluteError((1, 1))
if scaler.scale_ is not None:
    mae_metric.set_scale(np.expand_dims(scaler.scale_, axis=0))


model = make_painn(optimizer, mae_metric)

# Start training
hist = model.fit(xtrain, ytrain,
                    epochs=epo,
                    callbacks=cbks,
                    validation_freq=epostep,
                    validation_data=(xtest, ytest),
                    verbose=2
                    )

#Calculate MAE
pred_val = model.predict(xval)
pred_val = scaler.inverse_transform(pred_val)
pred_train = model.predict(xtrain)
pred_train = scaler.inverse_transform(pred_train)
true_train = scaler.inverse_transform(ytrain)
true_val = scaler.inverse_transform(yval)

mae_valid = np.mean(np.abs(true_val-pred_val ))

print("Training finished, here is your MAE: ",mae_valid)
