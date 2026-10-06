import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from sklearn.preprocessing import StandardScaler
import os

MODEL_PATH = "lstm_autoencoder.keras"

class EnvironmentalAnomalyDetector:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.threshold = 0.0

    def generate_synthetic_training_data(self, samples=1000, time_steps=1, features=5):
        """
        Generates normal synthetic baseline data (simulating normal weather/AQI) 
        so we can train the autoencoder to know what 'normal' looks like.
        Features: pm10, pm2_5, carbon_monoxide, nitrogen_dioxide, ozone
        """
        print("Generating baseline training data...")
        # Simulating normal ranges
        data = np.random.normal(loc=[20, 15, 200, 30, 0.5], scale=[5, 3, 50, 10, 0.2], size=(samples, features))
        # Ensure no negative values
        data = np.clip(data, 0, None)
        return data

    def build_model(self, time_steps, features):
        """
        Builds the advanced LSTM Autoencoder Neural Network.
        """
        print("Building LSTM Autoencoder Architecture...")
        model = Sequential([
            # Encoder
            LSTM(32, activation='relu', input_shape=(time_steps, features), return_sequences=False),
            RepeatVector(time_steps),
            # Decoder
            LSTM(32, activation='relu', return_sequences=True),
            TimeDistributed(Dense(features))
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def train(self):
        """
        Trains the model on normal data and calculates the anomaly threshold.
        """
        X_train_raw = self.generate_synthetic_training_data()
        
        # Scale the data
        X_train_scaled = self.scaler.fit_transform(X_train_raw)
        
        # Reshape for LSTM [samples, time_steps, features]
        time_steps = 1
        features = X_train_scaled.shape[1]
        X_train = X_train_scaled.reshape((X_train_scaled.shape[0], time_steps, features))

        self.model = self.build_model(time_steps, features)
        
        print("Training the Deep Learning Model... (This will take a few seconds)")
        self.model.fit(X_train, X_train, epochs=15, batch_size=32, validation_split=0.1, verbose=0)
        
        # Calculate Threshold (Max error on normal data)
        X_pred = self.model.predict(X_train, verbose=0)
        train_mae_loss = np.mean(np.abs(X_pred - X_train), axis=2)
        # Set threshold high enough to not catch normal fluctuations, but catch large spikes
        self.threshold = np.max(train_mae_loss) * 1.5 
        
        print(f"Model trained! Anomaly Threshold set to: {self.threshold:.4f}")
        
        # Save the model
        self.model.save(MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

    def load(self):
        """Loads the pre-trained model."""
        if os.path.exists(MODEL_PATH):
            self.model = load_model(MODEL_PATH)
            # Need to fit scaler on synthetic data again just to load the mean/std parameters
            X_dummy = self.generate_synthetic_training_data(samples=100)
            self.scaler.fit(X_dummy)
            self.threshold = 0.5  # Approximate fallback threshold if loading dynamically
            print("Model loaded successfully.")
        else:
            raise FileNotFoundError("Model not found. Run train() first.")

    def detect_anomaly(self, live_data_dict):
        """
        Takes a single live data point, scales it, passes it through the LSTM, 
        and flags if the reconstruction error is too high.
        """
        features = ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "ozone"]
        values = [[live_data_dict[f] for f in features]]
        
        # Scale
        scaled_values = self.scaler.transform(values)
        
        # Reshape for LSTM
        X_live = scaled_values.reshape(1, 1, len(features))
        
        # Predict (Reconstruct)
        X_pred = self.model.predict(X_live, verbose=0)
        
        # Calculate Error
        mae_loss = np.mean(np.abs(X_pred - X_live), axis=2)[0][0]
        
        is_anomaly = mae_loss > self.threshold
        return is_anomaly, mae_loss

if __name__ == "__main__":
    detector = EnvironmentalAnomalyDetector()
    detector.train()
