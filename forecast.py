import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

#Importing the datasets
gen_1 = pd.read_csv('C:/Users/Md Faris/Desktop/ML Project/Development-work/Plant-1/Plant_1_Generation_Data.csv')
sens_1 = pd.read_csv('C:/Users/Md Faris/Desktop/ML Project/Development-work/Plant-1/Plant_1_Weather_Sensor_Data.csv')
gen_2 = pd.read_csv('C:/Users/Md Faris/Desktop/ML Project/Development-work/Plant-2/Plant_2_Generation_Data.csv')
sens_2 = pd.read_csv('C:/Users/Md Faris/Desktop/ML Project/Development-work/Plant-2/Plant_2_Weather_Sensor_Data.csv')

df = gen_1.copy()
df2= sens_1.copy()
df2['DATE_TIME'] = pd.to_datetime(sens_1['DATE_TIME'], format='%Y-%m-%d %H:%M:%S')

# Step 1: Convert 'DATE_TIME' to a datetime object
df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'], format='%d-%m-%Y %H:%M')

# Step 2: Set 'DATE_TIME' as the index
df.set_index('DATE_TIME', inplace=True)
df2.set_index('DATE_TIME', inplace=True)

sens_numeric = df2[['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']]
# Step 3: Resample the data into 15-minute intervals and sum the values over each interval
sens_resampled = sens_numeric.resample('15T').mean()
df_resampled = df.resample('15T').sum()

df_final = pd.merge(df_resampled, sens_resampled[['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']], 
                    left_index=True, right_index=True, how='inner')


df_final['DAILY_YIELD_MW'] = df_final['DAILY_YIELD']/1000
daily_yield = df_final['DAILY_YIELD_MW']
# Making a function to execute above idea

def df_to_X_y(df, window_size=5):  # 5 here means we are taking the last 5 values
    df_as_np = df.to_numpy() #converting the dataframe to numpy arrays
    X = []
    y = []
    for i in range(len(df_as_np)-window_size):
        row = [[a] for a in df_as_np[i:i+window_size]]
        X.append(row)
        label = df_as_np[i+window_size]
        y.append(label)
    return np.array(X), np.array(y)

WINDOW_SIZE = 5
X, y = df_to_X_y(daily_yield, WINDOW_SIZE)
X.shape, y.shape


# making a train and test dataset to train the model

X_train, y_train  = X[:2600], y[:2600]
X_val, y_val  = X[2600:2900], y[2600:2900]
X_test, y_test  = X[2900:], y[2900:]
#  Tensorflow imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, InputLayer, Dropout

# Defining the model and adding layers into the LSTM model
model1 = Sequential()

# Assuming input shape of (5, 1) for 60 minutes of daily yield data
model1.add(InputLayer(input_shape=(5, 1)))  # 5 time steps, 1 feature

# LSTM layer with 32 units, reduced from 64 to prevent overfitting
model1.add(LSTM(32))

# Adding a Dense layer with 8 units and ReLU activation
model1.add(Dense(8, activation='relu'))

# Adding Dropout for regularization (optional, 0.2 means 20% dropout rate)
model1.add(Dropout(0.2))

# Output layer for a single value prediction with linear activation
model1.add(Dense(1, activation='linear'))

# Compile the model
model1.compile(optimizer='adam', loss='mse')

# Model summary to check the architecture
model1.summary()

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError

# Define the model checkpoint callback
# Save the best model based on validation loss (lowest RMSE)
cp = ModelCheckpoint(
    filepath='model1/best_model.keras',  # Use .h5 extension for model saving
    monitor='val_root_mean_squared_error',  # Monitor validation RMSE
    save_best_only=True,
    save_weights_only=False,  # Save full model instead of just weights
    verbose=1  # Print out saving messages
)

# Add early stopping to stop training if the model stops improving
es = EarlyStopping(
    monitor='val_root_mean_squared_error',  # Stop if RMSE doesn't improve
    patience=10,  # Wait 10 epochs before stopping
    restore_best_weights=True,  # Restore the best weights after stopping
    verbose=1
)

# Compile the model
# Lower learning rate is set to help the optimizer converge more smoothly
model1.compile(
    loss=MeanSquaredError(),
    optimizer=Adam(learning_rate=1e-4),  # 0.0001 is written as 1e-4 for clarity
    metrics=[RootMeanSquaredError()]  # RMSE metric for model evaluation
)

# Fit the model with better structure, additional callbacks, and dynamic batch size
model1.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),  # Provide validation data for real-time evaluation
    epochs=75,  # Increase epochs for a more robust model, can be adjusted later
    batch_size=32,  # Use a moderate batch size (adjust based on your hardware)
    callbacks=[cp, es],  # Include EarlyStopping for better convergence
    verbose=1  # Show training progress
)


from tensorflow.keras.models import load_model

# Load the best saved model
model1 = load_model('model1/best_model.keras')  # Use the correct file extension and path

# Print model summary to confirm successful loading
model1.summary()


# Make predictions on the training data
train_predictions = model1.predict(X_train).flatten()

# Create a DataFrame to compare predictions and actual values
train_results = pd.DataFrame({
    'Train Predictions': train_predictions,
    'Actual Values': y_train.flatten()  # Ensure y_train is also flattened for consistency
})

import matplotlib.pyplot as plt

# Set plot size for better visibility
plt.figure(figsize=(10, 6))

# Plot Train Predictions and Actual Values
plt.plot(train_results['Train Predictions'][50:100], label='Train Predictions', color='blue', linestyle='--', marker='o')
plt.plot(train_results['Actual Values'][50:100], label='Actual Values', color='green', linestyle='-', marker='x')

# Add labels, title, and grid
plt.title('Comparison of Train Predictions vs Actual Values', fontsize=16)
plt.xlabel('Time Steps (Sample Index)', fontsize=12)
plt.ylabel('Daily Yield', fontsize=12)
plt.grid(True)

# Add a legend to differentiate the lines
plt.legend()

# Show the plot
# plt.tight_layout()
# plt.show()


# Make predictions on the validation data
val_predictions = model1.predict(X_val).flatten()

# Create a DataFrame to compare predictions and actual values
val_results = pd.DataFrame({
    'Val Predictions': val_predictions,
    'Actual Values': y_val.flatten()  # Ensure y_val is flattened for consistency
})
import matplotlib.pyplot as plt

# Set plot size for better visibility
plt.figure(figsize=(10, 6))

# Plot Validation Predictions and Actual Values
plt.plot(val_results['Val Predictions'][:50], label='Validation Predictions', color='blue', linestyle='--', marker='o')
plt.plot(val_results['Actual Values'][:50], label='Actual Values', color='orange', linestyle='-', marker='x')

# Add labels, title, and grid
plt.title('Comparison of Validation Predictions vs Actual Values', fontsize=16)
plt.xlabel('Time Steps (Sample Index)', fontsize=12)
plt.ylabel('Daily Yield', fontsize=12)
plt.grid(True)

# Add a legend to differentiate the lines
plt.legend()

# Show the plot
# plt.tight_layout()
# plt.show()


# Make predictions on the test data
test_predictions_1 = model1.predict(X_test).flatten()

# Create a DataFrame to compare predictions and actual values
test_results_1 = pd.DataFrame({
    'Test Predictions': test_predictions_1,
    'Actual Values': y_test.flatten()  # Ensure y_test is flattened for consistency
})

import matplotlib.pyplot as plt

# Set plot size for better visibility
plt.figure(figsize=(10, 6))

# Plot Test Predictions and Actual Values
plt.plot(test_results_1['Test Predictions'][:50], label='Test Predictions', color='blue', linestyle='--', marker='o')
plt.plot(test_results_1['Actual Values'][:50], label='Actual Values', color='red', linestyle='-', marker='x')

# Add labels, title, and grid
plt.title('Comparison of Test Predictions vs Actual Values', fontsize=16)
plt.xlabel('Time Steps (Sample Index)', fontsize=12)
plt.ylabel('Daily Yield', fontsize=12)
plt.grid(True)

# Add a legend to differentiate the lines
plt.legend()

# Show the plot
# plt.tight_layout()
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd

def evaluate_model_performance(y_true, y_pred):
    """
    Evaluates model performance by computing key metrics and generating visualizations.

    Parameters:
    y_true (array-like): Actual values
    y_pred (array-like): Predicted values

    Returns:
    dict: A dictionary containing calculated metrics (MSE, RMSE, MAE, R²)
    """

    # Flattening inputs if they are not 1D
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # Calculating key metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Printing metrics
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")

    # Residuals calculation
    residuals = y_true - y_pred

    # Plotting residuals distribution
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(residuals, bins=20, edgecolor='black')
    plt.title('Residuals Distribution')
    plt.xlabel('Residual')
    plt.ylabel('Frequency')

    # Plotting Predictions vs Actual Values
    plt.subplot(1, 2, 2)
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.title('Predictions vs Actual Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--')  # Ideal line

    # plt.tight_layout()
    # plt.show()

    # Plotting actual vs predicted over time (for time series or sequential data)
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label='Actual Values')
    plt.plot(y_pred, label='Predicted Values', alpha=0.7)
    plt.title('Actual vs Predicted Values Over Time')
    plt.xlabel('Index')
    plt.ylabel('Values')
    # plt.legend()
    # plt.show()

    # Returning a dictionary of computed metrics
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    }

    return metrics

train_metrics_model1 = evaluate_model_performance(y_test, test_predictions_1)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Define Model
model2 = Sequential()

# Stacked LSTM Layers with Dropout
model2.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model2.add(Dropout(0.2))
model2.add(LSTM(64, return_sequences=False))
model2.add(Dropout(0.2))

# Dense Layers
model2.add(Dense(32, activation='relu'))
model2.add(Dense(1, activation='linear'))

# Compile the model
model2.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['RootMeanSquaredError'])

# Set callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('model2_best.keras', save_best_only=True, monitor='val_loss')
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)

# Fit the model
history = model2.fit(X_train, y_train,
                     validation_data=(X_val, y_val),
                     epochs=100,
                     batch_size=64,
                     callbacks=[early_stopping, checkpoint, lr_scheduler])

# Load the best model weights
model2.load_weights('model2_best.keras')

# Print model summary to confirm successful loading
model2.summary()


# Make predictions on the test data
test_predictions_2 = model2.predict(X_test).flatten()

# Create a DataFrame to compare predictions and actual values
test_results_2 = pd.DataFrame({
    'Test Predictions': test_predictions_2,
    'Actual Values': y_test.flatten()  # Ensure y_test is flattened for consistency
})
test_results_2.to_csv('csv/comparison_results.csv', index=False)

import matplotlib.pyplot as plt

# Set plot size for better visibility
plt.figure(figsize=(10, 6))

# Plot Test Predictions and Actual Values
plt.plot(test_results_2['Test Predictions'][:50], label='Test Predictions', color='blue', linestyle='--', marker='o')
plt.plot(test_results_2['Actual Values'][:50], label='Actual Values', color='red', linestyle='-', marker='x')

# Add labels, title, and grid
plt.title('Comparison of Test Predictions vs Actual Values', fontsize=16)
plt.xlabel('Time Steps (Sample Index)', fontsize=12)
plt.ylabel('Daily Yield', fontsize=12)
plt.grid(True)

# Add a legend to differentiate the lines
plt.legend()

# Show the plot
# plt.tight_layout()
# plt.show()

train_metrics_model2 = evaluate_model_performance(y_test, test_predictions_2)






from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def predict_future_seq2seq(model, data, n_future_steps, window_size=5):
  all_predictions = []

  # Iterate through the data with a sliding window
  for i in range(len(data) - window_size + 1):
    # Get the current window of data
    input_data = data[i:i + window_size]

    # Reshape the data for the LSTM model (batch size = 1, window size, features = 1)
    input_data_reshaped = input_data.reshape((1, window_size, 1))

    # Predict the next step using the model
    next_prediction = model.predict(input_data_reshaped)

    # Flatten the prediction and append it to the all_predictions list
    next_value = next_prediction.flatten()[0]
    all_predictions.append(next_value)

  # Select the last n_future_steps predictions for future forecast
  future_predictions = all_predictions[-n_future_steps:]

  return future_predictions

# Get the dataset as numpy array
data = daily_yield.to_numpy()

# Reduce the number of future 15-minute intervals for faster prediction
n_future_intervals = 3 * 96  # Predict only the next 10 intervals (15 minutes each)

# Forecast future values using the trained model in a single step for sequence-to-sequence
future_predictions = predict_future_seq2seq(model2, data, n_future_steps=n_future_intervals)

# Create future date range based on the last date in the dataset, spaced by 15 minutes
last_date = df_final.index[-1]
future_dates = [last_date + timedelta(minutes=15 * i) for i in range(n_future_intervals)]

# # Ensure future_predictions and future_dates have the same length
# # Truncate or pad the lists to match the shorter one if needed
# min_length = min(len(future_predictions), len(future_dates))
# future_predictions = future_predictions[:min_length]
# future_dates = future_dates[:min_length]

# Create a DataFrame for future predictions
future_df = pd.DataFrame({
    'DATE_TIME': future_dates,
    'Predicted_Power': future_predictions
})
future_df.to_csv('csv/future_predictions.csv', index=False)
# Plot the future predictions
plt.figure(figsize=(10, 6))
plt.plot(df_resampled.index[-100:], daily_yield[-100:], label='Actual Power', color='blue')
plt.plot(future_df['DATE_TIME'], future_df['Predicted_Power'], label='Predicted Future Power (15-min intervals)', color='red', linestyle='--', marker='o')

# Add labels, title, and grid
plt.title('Solar Power Forecast for Future Intervals', fontsize=16)
plt.xlabel('Date and Time', fontsize=12)
plt.ylabel('Power (MW)', fontsize=12)
plt.grid(True)

# Add a legend
plt.legend()

# Save the plot as an image
plt.tight_layout()
plt.savefig('images/forecast.png')
