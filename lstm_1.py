import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau
import typing_extensions

data = pd.read_csv("Data.csv")

# Convert 'Time' column to datetime format
data["Time"] = pd.to_datetime(data["Time"])

# Normalize 'Time' column
data["time_normalized"] = (data['Time'] - data['Time'].min()) / (data['Time'].max() - data['Time'].min())

data = data.drop(["Time"], axis=1)

# Normalize other columns
scaler = MinMaxScaler()
data[['Temperature', 'AxialAxisRmsVibration', 'RadialAxisKurtosis', 'RadialAxisPeakAcceleration',
      'RadialAxisRmsAcceleration', 'RadialAxisRmsVibration']] = scaler.fit_transform(data[['Temperature',
                                                                                           'AxialAxisRmsVibration',
                                                                                           'RadialAxisKurtosis',
                                                                                           'RadialAxisPeakAcceleration',
                                                                                           'RadialAxisRmsAcceleration',
                                                                                           'RadialAxisRmsVibration']])

split_index_train = int(len(data) * 0.8)  # 80% for training
split_index_val = int(len(data) * 0.9)  # 10% for validation, 10% for testing

train_data = data.iloc[:split_index_train]
val_data = data.iloc[split_index_train:split_index_val]
test_data = data.iloc[split_index_val:]

# Sequence length for TimeseriesGenerator
sequence_length = 500

# TimeseriesGenerator for training
train_sequence_gen = TimeseriesGenerator(data=train_data[['time_normalized']].values,
                                         targets=train_data['Temperature'].values,
                                         length=sequence_length, batch_size=128)

# TimeseriesGenerator for validation
val_sequence_gen = TimeseriesGenerator(data=val_data[['time_normalized']].values,
                                       targets=val_data['Temperature'].values,
                                       length=sequence_length, batch_size=128)

# TimeseriesGenerator for testing
test_sequence_gen = TimeseriesGenerator(data=test_data[['time_normalized']].values,
                                        targets=test_data['Temperature'].values,
                                        length=sequence_length, batch_size=128)

# Retrieve batches from the generators
X_train, y_train = train_sequence_gen[0]
X_val, y_val = val_sequence_gen[0]
X_test, y_test = test_sequence_gen[0]


def create_model(inputShape, epochs, batch_size):
    model = Sequential([
        LSTM(units=100, return_sequences=True, input_shape=inputShape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50),
        Dropout(0.2),
        Dense(units=1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_val, y_val), callbacks=[reduce_lr])

    loss, mae = model.evaluate(X_test, y_test)
    print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
    print("settings--> Epoch:", epochs, "|| Batch_size:", batch_size, "|| Sequence len.:", sequence_length)
    print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
    print(f"Test Loss(mse): {loss}")
    print(f"Test MAE: {mae}")
    print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")

    test_losses = [loss] * len(history.history['loss'])
    l = [loss]

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.plot(test_losses)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation', 'Test'], loc='upper left')
    plt.title("LSTM( LOSS VS. EPOCH )")
    plt.show()

    return model, l


model_1, L = create_model((sequence_length, 1), 2, 16)

predictions = model_1.predict(X_test)
print(model_1.summary())
plt.scatter(range(len(y_test)), y_test, label='Actual', color='red')
plt.scatter(range(len(predictions)), predictions, label='predictions')
plt.title("LSTM( ACTUAL VS. PRED. )")
plt.legend()
plt.show()

"""
list = []
for x in range(10):
    model, l = create_model((sequence_length, 1), 40, 64)
    list.append(l)

plt.ylabel("TEST LOSS")
plt.title("LSTM LOSS CHANGE IN SAME SETTINGS")
plt.plot(list)
plt.show()
"""
