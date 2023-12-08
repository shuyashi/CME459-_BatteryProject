## using feature set 2 to generate the LR model
from scipy import io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Loading the data in python
data = io.loadmat('feature_set2.mat')
print(data['x_matrix'].shape)
print(data['y_vector'].shape)
# x_matrix = np.concatenate([np.ones((len(data['x_matrix']),1)),np.array(data['x_matrix'])],1)
x_matrix = np.array(data['x_matrix'])
y_vector = np.array(data['y_vector'])

# add normalization step to the data
scaler = StandardScaler()
x_normalization = scaler.fit_transform(x_matrix)
y_normalization = scaler.fit_transform(y_vector)

## following method if not using the LinearRegression function from python
# x_train = np.concatenate([np.ones((len(x_train),1)),np.array(x_train)],1)
# x_test = np.concatenate([np.ones((len(x_test),1)),np.array(x_test)],1)
# x_secondary_test = np.concatenate([np.ones((len(x_secondary_test),1)),np.array(x_secondary_test)],1)
# y_train = np.array(y_train)
# y_test = np.array(y_test)
# y_secondary_test = np.array(y_secondary_test)

# split train, test data, using splitting function
x_train,x_test,y_train,y_test = train_test_split(x_normalization,y_normalization,test_size=0.2)
# test_size is the 20% of the data set to be used as test set
print(x_train.shape)
print(x_test.shape)

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

# plot for fitting the train and predicted values with scaling
fig = plt.figure(4)
ax1 = fig.add_subplot(111)
x_plot = np.arange(0,98,1)
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
x_plot_test = np.arange(0,25,1)
ax1.scatter(x_plot_test, y_test, label='Test Value')
ax1.scatter(x_plot_test, y_hat_test, label='Predicted Value')
plt.legend(['Test Values', 'Predicted values'])
plt.xlabel('X_test')
plt.ylabel('Y_test')
plt.show()

# remove scaling to do the plot
Y_train = y_train * np.std(y_vector) + np.mean(y_vector)
Y_test = y_test * np.std(y_vector) + np.mean(y_vector)
Y_hat_train = y_hat_train * np.std(y_vector) + np.mean(y_vector)
Y_hat_test = y_hat_test * np.std(y_vector) + np.mean(y_vector)

# Calculate RMSE of train
rmse_train = np.sqrt(mean_squared_error(Y_train, Y_hat_train))
print("RMSE of train:", rmse_train)

# Calculate RMSE of test
rmse_test = np.sqrt(mean_squared_error(Y_test, Y_hat_test))
print("RMSE of test:", rmse_test)

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

# Create a scatter plot of standardized residuals against predicted values of training, should be [-3,3]
plt.scatter(Y_hat_train, standarized_resi_train, alpha=0.5)
plt.xlabel("Predicted Values")
plt.ylabel("Standardized Residuals")
plt.title("Standardized Residual Plot of training")
plt.axhline(y=0, color='r', linestyle='--')
plt.show()

# Create a figure
plt.figure(figsize=(8, 6))
# Scatter plot for Y_train vs. Y_hat_train
plt.scatter(Y_train, Y_hat_train, color='blue', label='Train (Actual vs. Predicted)')
# Scatter plot for Y_test vs. Y_hat_test
plt.scatter(Y_test, Y_hat_test, color='red', label='Test (Actual vs. Predicted)')
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