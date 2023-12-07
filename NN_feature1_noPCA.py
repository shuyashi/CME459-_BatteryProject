import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy import io

# Loading the data in python
data = io.loadmat('feature_set1.mat')
print(data['x_matrix'].shape)
print(data['y_vector'].shape)
x_matrix = np.array(data['x_matrix'])
y_vector = np.array(data['y_vector'])

# Normalization step
scaler = StandardScaler()
x_normalization = scaler.fit_transform(x_matrix)
y_normalization = scaler.fit_transform(y_vector)
x_sec_normalization = scaler.fit_transform(x_matrix)

# split train, test, secondary test data, defined from the paper
numBat1 = 41
numBat2 = 42  # Different from the paper (43), since the b2c1 is deleted
numBat3 = 40
numBat = numBat1 + numBat2 + numBat3

test_ind = np.hstack((np.arange(0, (numBat1 + numBat2), 2), 83))
print(test_ind)
train_ind = np.arange(1, (numBat1 + numBat2 - 1), 2)
print(train_ind)
secondary_test_ind = np.arange(numBat - numBat3 - 1, numBat)
print(secondary_test_ind)
x_train = x_normalization[train_ind, :]
x_test = x_normalization[test_ind, :]
x_secondary_test = x_normalization[secondary_test_ind, :]

y_train = y_normalization[train_ind, :]
y_test = y_normalization[test_ind, :]
y_secondary_test = y_normalization[secondary_test_ind, :]

# Create the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(50, activation='relu', input_shape=(6,), bias_initializer='zeros'),
    tf.keras.layers.Dense(30, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
model.compile(optimizer=optimizer, loss='MeanSquaredError', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=500, batch_size=40, verbose=1)

# model on the train set
y_model_pred_train = model.predict(x_train)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
y_model_pred_test = model.predict(x_test)
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)

# Evaluate the model on the secondary test set
loss_sec, accuracy_sec = model.evaluate(x_secondary_test, y_secondary_test, verbose=0)
y_model_pred_sec_test = model.predict(x_secondary_test)
print('Test secondary Loss:', loss_sec)
print('Test secondary Accuracy:', accuracy_sec)

# remove scaling to do the plot
Y_train = y_train * np.std(y_vector) + np.mean(y_vector)
Y_test = y_test * np.std(y_vector) + np.mean(y_vector)
Y_secondary_test = y_secondary_test * np.std(y_vector) + np.mean(y_vector)
Y_model_pred_train = y_model_pred_train * np.std(y_vector) + np.mean(y_vector)
Y_model_pred_test = y_model_pred_test * np.std(y_vector) + np.mean(y_vector)
Y_model_pred_secondary_test = y_model_pred_sec_test * np.std(y_vector) + np.mean(y_vector)

# Calculate RMSE of test
rmse_train = np.sqrt(mean_squared_error(Y_train, Y_model_pred_train))
print("RMSE of train:", rmse_train)

# Calculate RMSE of test
rmse_test = np.sqrt(mean_squared_error(Y_test, Y_model_pred_test))
print("RMSE of test:", rmse_test)

# Calculate RMSE of secondary test
rmse_sec_test = np.sqrt(mean_squared_error(Y_secondary_test, Y_model_pred_secondary_test))
print("RMSE of secondary test:", rmse_sec_test)

# Calculate the absolute percentage error for testing
percentage_errors_test = np.abs((Y_test - Y_model_pred_test) / Y_test) * 100
# Calculate the Mean Percentage Error (MPE)
mpe_test = np.mean(percentage_errors_test)
print("Mean Percentage Error (MPE) of testing:", mpe_test)

# Calculate the absolute percentage error for secondary testing
percentage_errors_sec_test = np.abs((Y_secondary_test - Y_model_pred_secondary_test) / Y_secondary_test) * 100
# Calculate the Mean Percentage Error (MPE)
mpe_sec_test = np.mean(percentage_errors_sec_test)
print("Mean Percentage Error (MPE) of secondary testing:", mpe_sec_test)

# plot for testing
plt.plot(Y_test, label='Actual data')
plt.plot(Y_model_pred_test, label='testing predictions')
# Set labels and title
plt.xlabel('Time step')
plt.ylabel('Cycle Life')
plt.title('Actual Vs. prediction')
# Display the legend
plt.legend()
# Display the plot
plt.show()

# plot for secondary testing
plt.plot(Y_secondary_test, label='Actual data')
plt.plot(Y_model_pred_secondary_test, label='secondary testing predictions')
# Set labels and title
plt.xlabel('Time step')
plt.ylabel('Cycle Life')
plt.title('Actual Vs. prediction')
# Display the legend
plt.legend()
# Display the plot
plt.show()

# Neural Network Model
# Extract the model weights
model_weights = []
for layer in model.layers:
    layer_weights = layer.get_weights()
    model_weights.append(layer_weights)


# Define the activation functions
def tanh(x):
    return np.tanh(x)


def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


# Define the model prediction function with training
def predict(x):
    layer1_weights, layer2_weights, layer3_weights = model_weights[0], model_weights[1], model_weights[2]
    layer1_output = relu(np.dot(x, layer1_weights[0]) + layer1_weights[1])
    layer2_output = relu(np.dot(layer1_output, layer2_weights[0]) + layer2_weights[1])
    layer3_output = np.dot(layer2_output, layer3_weights[0]) + layer3_weights[1]
    return layer3_output

    # Perform model prediction on X_train
    predictions_train = [predict(x) for x in x_train]

    return predictions_train

# Define the model prediction function with testing
def predict_test(x):
    layer1_weights, layer2_weights, layer3_weights = model_weights[0], model_weights[1], model_weights[2]
    layer1_output = relu(np.dot(x, layer1_weights[0]) + layer1_weights[1])
    layer2_output = relu(np.dot(layer1_output, layer2_weights[0]) + layer2_weights[1])
    layer3_output = np.dot(layer2_output, layer3_weights[0]) + layer3_weights[1]
    return layer3_output

    # Perform model prediction on X_test
    predictions_test = [predict(x) for x in x_test]

    return predictions_test

# Define the model prediction function with secondary testing
def predict_sec(x):
    layer1_weights, layer2_weights, layer3_weights = model_weights[0], model_weights[1], model_weights[2]
    layer1_output = relu(np.dot(x, layer1_weights[0]) + layer1_weights[1])
    layer2_output = relu(np.dot(layer1_output, layer2_weights[0]) + layer2_weights[1])
    layer3_output = np.dot(layer2_output, layer3_weights[0]) + layer3_weights[1]
    return layer3_output

    # Perform model prediction on X_secondary_test
    predictions_sec = [predict_sec(x) for x in x_secondary_test]

    return predictions_sec

# Get the model predictions for x_test
y_pred_test = predict(x_test)
Y_pred_test = y_pred_test * np.std(y_vector) + np.mean(y_vector)
plt.plot(Y_model_pred_test, ":r", label='inbuilt')
plt.plot(Y_pred_test, "b.", label='explicit_model')
# Set labels and title
plt.xlabel('Time step')
plt.ylabel('Cycle Life')
plt.title('Inbuilt_prediction Vs. Explicit_model_predictions testing')
# Display the legend
plt.legend()
# Display the plot
plt.show()

# Get the model predictions for x_secondary_test
y_pred_sec_test = predict_sec(x_secondary_test)
Y_pred_sec_test = y_pred_sec_test * np.std(y_vector) + np.mean(y_vector)
plt.plot(Y_model_pred_secondary_test, ":r", label='inbuilt')
plt.plot(Y_pred_sec_test, "b.", label='explicit_model')
# Set labels and title
plt.xlabel('Time step')
plt.ylabel('Cycle Life')
plt.title('Inbuilt_prediction Vs. Explicit_model_predictions secondary testing')
# Display the legend
plt.legend()
# Display the plot
plt.show()

# print the model summary, the model function is y = w_i * x + b_i
print(model.summary())

# Extract the model configuration
model_config = model.get_config()
# Create a new model from the extracted configuration
new_model = tf.keras.models.Sequential.from_config(model_config)