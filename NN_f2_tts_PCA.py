import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy import io
from sklearn.model_selection import train_test_split

# Loading the data in python
data = io.loadmat('feature_set2.mat')
print(data['x_matrix'].shape)
print(data['y_vector'].shape)
x_matrix = np.array(data['x_matrix'])
y_vector = np.array(data['y_vector'])

# Normalization step
scaler = StandardScaler()
x_normalization = scaler.fit_transform(x_matrix)
y_normalization = scaler.fit_transform(y_vector)
x_sec_normalization = scaler.fit_transform(x_matrix)

# Perform PCA for dimensionality reduction
pca = PCA()  # Do not specify the number of components initially
X_pca = pca.fit_transform(x_normalization)

# Determine the number of components to keep based on explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
num_components = np.argmax(cumulative_variance_ratio >= 0.95) + 1  # Keep components explaining 95% variance

# Access eigenvalues
eigenvalues = pca.explained_variance_
# Sort eigenvalues in descending order
sorted_eigenvalues = np.sort(eigenvalues)[::-1]
# Print eigenvalues
print(sorted_eigenvalues)

# Plot the elbow plot, choose the largest few PCs to train the model which has the highest valiability
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Elbow Plot for PCA')
plt.show()

# Apply PCA with the selected number of components
pca = PCA(n_components=num_components)
X_pca = pca.fit_transform(x_normalization)

# Convert the reduced PC set back to the original set and check the accuracy
# X_pca_reconstructed = np.dot(X_pca, pca.components_) + pca.mean_
X_pca_reconstructed = pca.inverse_transform(X_pca)

# # Checking accuracy: accuracy_score function from scikit-learn does not support multi-output or continuous targets.
# # It is typically used for classification tasks with discrete labels.
# # If you have continuous targets or multi-output regression, you need to use a different evaluation metric.
# mse_train = mean_squared_error(X_train, X_train_reconstructed)
# print("Training set MSE: ", mse_train)

# plot for the reduced pc set and the original set
plt.figure(2)
plt.plot(X_pca[:, 3], label='Actual data')
plt.plot(X_pca_reconstructed[:, 1], label='PCA reconstruction')
# Set labels and title
plt.xlabel('Time step')
plt.title('Actual Vs. pca')
# Display the legend
plt.legend()
# Display the plot
plt.show()

# split train, test data, using splitting function
x_train,x_test,y_train,y_test = train_test_split(X_pca,y_normalization,test_size=0.2)
# test_size is the 20% of the data set to be used as test set

# Create the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(40, activation='relu', input_shape=(6,), bias_initializer='zeros'),
    tf.keras.layers.Dense(15, activation='relu'),
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


# remove scaling to do the plot
Y_train = y_train * np.std(y_vector) + np.mean(y_vector)
Y_test = y_test * np.std(y_vector) + np.mean(y_vector)
Y_model_pred_train = y_model_pred_train * np.std(y_vector) + np.mean(y_vector)
Y_model_pred_test = y_model_pred_test * np.std(y_vector) + np.mean(y_vector)

# Calculate RMSE of test
rmse_train = np.sqrt(mean_squared_error(Y_train, Y_model_pred_train))
print("RMSE of train:", rmse_train)

# Calculate RMSE of test
rmse_test = np.sqrt(mean_squared_error(Y_test, Y_model_pred_test))
print("RMSE of test:", rmse_test)

# Calculate the absolute percentage error for testing
percentage_errors_test = np.abs((Y_test - Y_model_pred_test) / Y_test) * 100
# Calculate the Mean Percentage Error (MPE)
mpe_test = np.mean(percentage_errors_test)
print("Mean Percentage Error (MPE) of testing:", mpe_test)

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

# print the model summary, the model function is y = w_i * x + b_i
print(model.summary())

# Extract the model configuration
model_config = model.get_config()
# Create a new model from the extracted configuration
new_model = tf.keras.models.Sequential.from_config(model_config)