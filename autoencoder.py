import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

# a. Import required libraries
# numpy, pandas, tensorflow.keras, sklearn

# b. Upload / access the dataset
# Assuming you have the credit card dataset loaded as a pandas DataFrame
df = pd.read_csv('path/to/credit_card_data.csv')

# Preprocess the data
X = df.drop('label', axis=1).values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# c. Encoder converts it into latent representation
input_layer = Input(shape=(X_scaled.shape[1],))
encoded = Dense(128, activation='relu')(input_layer)
encoded = Dropout(0.2)(encoded)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dropout(0.2)(encoded)
latent_repr = Dense(32, activation='relu')(encoded)

# d. Decoder networks convert it back to the original input
decoded = Dense(64, activation='relu')(latent_repr)
decoded = Dropout(0.2)(decoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dropout(0.2)(decoded)
output_layer = Dense(X_scaled.shape[1], activation='linear')(decoded)

# e. Compile the models with Optimizer, Loss, and Evaluation Metrics
autoencoder = Model(input_layer, output_layer)
autoencoder.compile(optimizer=Adam(lr=0.001), loss='mse', metrics=['mse'])

# Train the autoencoder
autoencoder.fit(X_scaled, X_scaled,
                epochs=100,
                batch_size=128,
                validation_split=0.2)

# Evaluate the autoencoder
_, mse = autoencoder.evaluate(X_scaled, X_scaled)
print(f'Mean Squared Error: {mse:.2f}')

# Use the autoencoder for anomaly detection
reconstruction = autoencoder.predict(X_scaled)
mse_per_sample = np.mean(np.power(X_scaled - reconstruction, 2), axis=1)

# Identify anomalies based on a threshold
threshold = np.mean(mse_per_sample) + 2 * np.std(mse_per_sample)
anomalies = df[mse_per_sample > threshold]
print(f'Number of anomalies detected: {len(anomalies)}')