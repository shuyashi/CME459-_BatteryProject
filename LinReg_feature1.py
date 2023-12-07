## using feature set 1 (discharge capacity) to generate the LR model
from scipy import io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.metrics import mean_squared_error

# Loading the data in python
data = io.loadmat('feature_set1.mat')
print(data['x_matrix'].shape)
print(data['y_vector'].shape)
# x_matrix = np.concatenate([np.ones((len(data['x_matrix']),1)),np.array(data['x_matrix'])],1)
x_matrix = np.array(data['x_matrix'])
y_vector = np.array(data['y_vector'])

# add normalization step to the data
scaler = StandardScaler()
x_normalization = scaler.fit_transform(x_matrix)
y_normalization = scaler.fit_transform(y_vector)

# use pca to reduce features
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
print("sorted_eigenvalues are:", sorted_eigenvalues)

# Plot the elbow plot, choose the largest few PCs to train the model which has the highest variability
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Elbow Plot for PCA')
plt.show()

# Apply PCA with the selected number of components
pca = PCA(n_components=num_components)
X_pca = pca.fit_transform(x_normalization)

# Convert the reduced PC set back to the original set and check the accuracy
X_pca_reconstructed = np.dot(X_pca, pca.components_) + pca.mean_
# X_sec_pca_reconstructed = np.dot(X_sec_pca, pca.components_) + pca.mean_
# plot for the reduced pc set and the original set for the entire x matrix
plt.figure(1)
plt.plot(X_pca[:, 0], label='Actual data')
plt.plot(X_pca_reconstructed[:, 0], label='PCA reconstruction')
# Set labels and title
plt.xlabel('Time step')
plt.title('Actual Vs. pca')
# Display the legend
plt.legend()
# Display the plot
plt.show()

# Histograms for feature 3, check distribution of it. Normal distribution is the best
# if not, either increasing the data point to approach, or using nonlinear reg model, or do normalization to the data
feature = 2  # Feature number (column number in x_matrix) for which to plot a histogram for
bins = 10  # No. of bins for histogram
plt.hist(x_matrix[:, feature], bins=bins)
plt.show()

# Correlation between output y and feature x (=1), to see the trend
# if not using pca, then checking the trend to reduce the features who have no strong linear correlation
feature_number = 1
plt.scatter(X_pca[:, feature_number], y_normalization)
plt.show()

## following method if not using the LinearRegression function from python
# x_train = np.concatenate([np.ones((len(x_train),1)),np.array(x_train)],1)
# x_test = np.concatenate([np.ones((len(x_test),1)),np.array(x_test)],1)
# x_secondary_test = np.concatenate([np.ones((len(x_secondary_test),1)),np.array(x_secondary_test)],1)
# y_train = np.array(y_train)
# y_test = np.array(y_test)
# y_secondary_test = np.array(y_secondary_test)

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
x_train = X_pca[train_ind, :]
x_test = X_pca[test_ind, :]
x_secondary_test = X_pca[secondary_test_ind, :]

y_train = y_normalization[train_ind, :]
y_test = y_normalization[test_ind, :]
y_secondary_test = y_normalization[secondary_test_ind, :]

# histogram of cycle life (y) for the training, test and secondary test datasets
fig = plt.figure(3)
plt.figure(figsize=(8, 6))
# Plot histogram for training
plt.hist(y_train, bins=20, alpha=0.5, color='red', label='Training')
# Plot histogram for testing
plt.hist(y_test, bins=20, alpha=0.5, color='green', label='Testing')
# Plot histogram for secondary test
plt.hist(y_secondary_test, bins=20, alpha=0.5, color='yellow', label='Secondary testing')
# Add labels and legend
plt.xlabel('Value')
plt.ylabel('Cycle life')
plt.title('Histogram Plot of cycle life with three datasets')
plt.legend()
# Show the plot
plt.show()

# Create a linear regression model
model = LinearRegression()

# Fit the model to the data
model.fit(x_train, y_train)

# Get the slope and intercept
slope = model.coef_[0]
intercept = model.intercept_

# Print the slope and intercept
print("Slope:", slope)
print("Intercept:", intercept)

# Predict y values using the linear regression model
y_hat_train = model.predict(x_train)
y_hat_test = model.predict(x_test)
y_hat_secondary_test = model.predict(x_secondary_test)

# Calculating SSE and SST
SSE_train = np.sum((y_train - y_hat_train) ** 2)
SST_train = np.sum((y_train - np.mean(y_train)) ** 2)
rsquared_train = 1 - (SSE_train / SST_train)
print("The R^2 of the train data is: ", rsquared_train)

# calculate SSE and SST for test and predicted values
SSE_test = np.sum((y_test - y_hat_test) ** 2)
SST_test = np.sum((y_test - np.mean(y_test)) ** 2)
rsquared_test = 1 - (SSE_test / SST_test)
print("The R^2 of the test data is: ", rsquared_test)

# SSE and SST for secondary test values
SSE_sec_test = np.sum((y_secondary_test - y_hat_secondary_test) ** 2)
SST_sec_test = np.sum((y_secondary_test - np.mean(y_secondary_test)) ** 2)
rsquared_sec_test = 1 - (SSE_sec_test / SST_sec_test)
print("The R^2 of the secondary test data is: ", rsquared_sec_test)

# Plotting scatter matrix directly using pandas, the matrix version of the data comparison
df = pd.DataFrame(x_train[:, 1:], columns=['Feature 1', 'Feature 2', 'Feature 3'])
scatter_matrix(df, alpha=0.2, figsize=(12, 12), diagonal='kde')
plt.show()

# plot for fitting the train and predicted values with scaling
fig = plt.figure(4)
ax1 = fig.add_subplot(111)
x_plot = np.arange(0, 41, 1)
ax1.scatter(x_plot, y_train, label='Actual Value')
ax1.scatter(x_plot, y_hat_train, label='Predicted Value')
# Plot the data and regression line
# plt.plot(x_plot, y_train,x_plot,y_hat_train)
plt.legend(['Actual Values', 'Predicted values'])
# plt.plot(x, y_pred, color='red', label='Linear regression')
plt.xlabel('X_train')
plt.ylabel('Y_train')
# plt.legend()
plt.show()

# plot the test values with the predicted value
fig = plt.figure(5)
ax1 = fig.add_subplot(111)
x_plot_test = np.arange(0, 43, 1)
ax1.scatter(x_plot_test, y_test, label='Test Value')
ax1.scatter(x_plot_test, y_hat_test, label='Predicted Value')
plt.legend(['Test Values', 'Predicted values'])
plt.xlabel('X_test')
plt.ylabel('Y_test')
plt.show()

# plot for fitting the train and predicted values with scaling
fig = plt.figure(6)
ax1 = fig.add_subplot(111)
x_plot_sec_test = np.arange(0, 41, 1)
ax1.scatter(x_plot_sec_test, y_secondary_test, label='Secondary test Value')
ax1.scatter(x_plot_sec_test, y_hat_secondary_test, label='Predicted Value')
# Plot the data and regression line
# plt.plot(x_plot, y_train,x_plot,y_hat_train)
plt.legend(['Secondary test Value', 'Predicted values'])
# plt.plot(x, y_pred, color='red', label='Linear regression')
plt.xlabel('X_secondary_test')
plt.ylabel('Y_secondary_test')
# plt.legend()
plt.show()

# remove scaling to do the plot
Y_train = y_train * np.std(y_vector) + np.mean(y_vector)
Y_test = y_test * np.std(y_vector) + np.mean(y_vector)
Y_secondary_test = y_secondary_test * np.std(y_vector) + np.mean(y_vector)
Y_hat_train = y_hat_train * np.std(y_vector) + np.mean(y_vector)
Y_hat_test = y_hat_test * np.std(y_vector) + np.mean(y_vector)
Y_hat_secondary_test = y_hat_secondary_test * np.std(y_vector) + np.mean(y_vector)

# Calculate RMSE of train
rmse_train = np.sqrt(mean_squared_error(Y_train, Y_hat_train))
print("RMSE of train:", rmse_train)

# Calculate RMSE of test
rmse_test = np.sqrt(mean_squared_error(Y_test, Y_hat_test))
print("RMSE of test:", rmse_test)

# Calculate RMSE of secondary test
rmse_sec_test = np.sqrt(mean_squared_error(Y_secondary_test, Y_hat_secondary_test))
print("RMSE of secondary test:", rmse_sec_test)

# Calculate the absolute percentage error for training
percentage_errors_train = np.abs((Y_train - Y_hat_train) / Y_train) * 100
# Calculate the Mean Percentage Error (MPE)
mpe_train = np.mean(percentage_errors_train)
print("Mean Percentage Error (MPE) of training:", mpe_train)

# Calculate the absolute percentage error for testing
percentage_errors_test = np.abs((Y_test - Y_hat_test) / Y_test) * 100
# Calculate the Mean Percentage Error (MPE)
mpe_test = np.mean(percentage_errors_test)
print("Mean Percentage Error (MPE) of testing:", mpe_test)

# Calculate the absolute percentage error for secondary testing
percentage_errors_sec_test = np.abs((Y_secondary_test - Y_hat_secondary_test) / Y_secondary_test) * 100
# Calculate the Mean Percentage Error (MPE)
mpe_sec_test = np.mean(percentage_errors_sec_test)
print("Mean Percentage Error (MPE) of secondary testing:", mpe_sec_test)

# standard deviation of the data
# std of training data
residuals_train = Y_train - Y_hat_train
std_train = np.std(residuals_train)
print("Standard Deviation of Training data: ", std_train)
# std of testing data
residuals_test = Y_test - Y_hat_test
standarized_resi_train = (residuals_train - np.mean(residuals_train)) / np.std(residuals_train)
std_test = np.std(residuals_test)
print("Standard Deviation of Testing data: ", std_test)
# std of secondary testing data
residuals_sec_test = Y_secondary_test - Y_hat_secondary_test
std_sec_test = np.std(residuals_sec_test)
print("Standard Deviation of Secondary Testing data: ", std_sec_test)

# Create a scatter plot of standardized residuals against predicted values of training, should be [-3,3]
plt.scatter(Y_hat_train, standarized_resi_train, alpha=0.5)
plt.xlabel("Predicted Values")
plt.ylabel("Standardized Residuals")
plt.title("Standardized Residual Plot of training")
plt.axhline(y=0, color='r', linestyle='--')
plt.show()

# # Create a figure with subplots
# fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
# # Plot for Y_train vs. Y_hat_train
# axes[0].scatter(Y_train, Y_hat_train, color='blue', label='Actual vs. Predicted')
# axes[0].plot([min(Y_train), max(Y_train)], [min(Y_train), max(Y_train)], color='black', linestyle='--', lw=2,
#              label='Ideal Line')
# axes[0].set_xlabel('Actual Values')
# axes[0].set_ylabel('Predicted Values')
# axes[0].set_title('Polarity Plot for Linear Regression (Train)')
# axes[0].legend()
# axes[0].grid(True)
# # Plot for Y_test vs. Y_hat_test
# axes[1].scatter(Y_test, Y_hat_test, color='green', label='Test vs. Predicted')
# axes[1].plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], color='black', linestyle='--', lw=2,
#              label='Ideal Line')
# axes[1].set_xlabel('Test Values')
# axes[1].set_ylabel('Predicted Values')
# axes[1].set_title('Polarity Plot for Linear Regression (Test)')
# axes[1].legend()
# axes[1].grid(True)
# # Plot for Y_secondary_test vs. Y_hat_secondary_test
# axes[2].scatter(Y_secondary_test, Y_hat_secondary_test, color='red', label='Secondary test vs. Predicted')
# axes[2].plot([min(Y_secondary_test), max(Y_secondary_test)], [min(Y_secondary_test), max(Y_secondary_test)], color='black', linestyle='--', lw=2,
#              label='Ideal Line')
# axes[2].set_xlabel('Secondary Test Values')
# axes[2].set_ylabel('Predicted Values')
# axes[2].set_title('Polarity Plot for Linear Regression (Secondary Test)')
# axes[2].legend()
# axes[2].grid(True)
# # Adjust spacing between subplots
# plt.tight_layout()
# # Show the merged plot
# plt.show()

# Create a figure
plt.figure(figsize=(8, 6))
# Scatter plot for Y_train vs. Y_hat_train
plt.scatter(Y_train, Y_hat_train, color='blue', label='Train (Actual vs. Predicted)')
# Scatter plot for Y_test vs. Y_hat_test
plt.scatter(Y_test, Y_hat_test, color='green', label='Test (Actual vs. Predicted)')
# Scatter plot for Y_secondary_test vs. Y_hat_secondary_test
plt.scatter(Y_secondary_test, Y_hat_secondary_test, color='red', label='Secondary Test (Actual vs. Predicted)')
# Plot the ideal line with the observed cycle life (y_vector)
plt.plot([min(y_vector), max(y_vector)], [min(y_vector), max(y_vector)], color='black', linestyle='-', lw=2,
         label='Ideal Line')
plt.xlabel('Experimental cycle life')
plt.ylabel('Predicted cycle life')
plt.title('Polarity Plot for Linear Regression Feature 1')
plt.legend()
plt.grid(True)
# Show the merged plot
plt.show()