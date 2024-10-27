import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical

def generate_sample_data():
    """
    Generates a mock dataset with 5 gestures, each containing 10 samples of 45 timesteps and 21 keypoints.
    """
    gestures = ["Alright", "Good Morning", "Good Afternoon", "Hello", "How are You"]
    num_gestures = len(gestures)
    samples_per_gesture = 10  # Number of samples per gesture
    timesteps = 45  
    keypoints = 21  

    # Generate mock data
    data = {}
    for gesture in gestures:
        data[gesture] = [
            np.random.rand(timesteps, keypoints).tolist() for _ in range(samples_per_gesture)
        ]
    
    X, y = [], []
    gesture_dict = {gesture: idx for idx, gesture in enumerate(gestures)}

    for gesture, samples in data.items():
        for sample in samples:
            X.append(sample)
            y.append(gesture_dict[gesture])

    X = np.array(X)
    y = np.array(y)
    
    # Convert labels to one-hot encoding
    y = to_categorical(y, num_classes=num_gestures)
    
    return X, y, gestures

def build_model(input_shape, num_classes):
    """
    Builds an LSTM model for gesture recognition.
    """
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=input_shape))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(256, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # Load sample data
    X, y, actions = generate_sample_data()
    
    # Verify shapes for input data
    print("X shape:", X.shape)  # Expected: (num_samples, 45, 21)
    print("y shape:", y.shape)  # Expected: (num_samples, num_gestures)

    # Define input shape
    input_shape = (X.shape[1], X.shape[2])  # (45, 21)
    
    # Build and train the model
    model = build_model(input_shape, len(actions))
    model.fit(X, y, epochs=5, batch_size=2)  # Adjust epochs and batch size as needed for actual training
    model.save('lstm_gesture_model.h5')
    
    print("Model training completed and saved as 'lstm_gesture_model.h5'.")


# model.save('lstm_gesture_model.keras')
