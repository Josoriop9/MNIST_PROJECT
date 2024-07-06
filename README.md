MNIST Digit Recognition
This project is a simple graphical application for recognizing handwritten digits using a pre-trained TensorFlow Keras model on the MNIST dataset. The application provides a drawing area where users can draw a digit, and the model predicts the digit in real-time.

Table of Contents
Installation
Usage
Features
Model Training
License
Installation
Prerequisites
Python 3.6 or later
virtualenv or venv for creating virtual environments
Steps
Clone the repository:

sh
Copy code
git clone https://github.com/Josoriop9/MNIST_PROJECT
cd MNIST_PROJECT
Create and activate a virtual environment:

On macOS and Linux:

sh
Copy code
python3 -m venv .venv
source .venv/bin/activate
On Windows:

sh
Copy code
python -m venv .venv
.venv\Scripts\activate
Install the required packages:

sh
Copy code
pip install -r requirements.txt
Ensure you have the trained model file mnist_model.keras in the project directory.

Usage
Run the application:

sh
Copy code
python app.py
Drawing and Predicting:

Use the mouse to draw a digit in the drawing area.
Click the "Predict" button to see the predicted digit.
Click the "Clear" button to clear the drawing area and start again.
Features
Real-Time Drawing: Draw digits on a 28x28 canvas.
Digit Prediction: Predict the drawn digit using a pre-trained MNIST model.
Clear Canvas: Clear the drawing area to draw and predict new digits.
Model Training
The model used in this project is pre-trained on the MNIST dataset. If you want to train your own model, you can use the following script (ensure TensorFlow is installed):

python
Copy code
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build the model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=200)

# Save the model
model.save('mnist_model.keras')
License
This project is licensed under the MIT License. See the LICENSE file for more details.

Make sure to include a requirements.txt file listing all necessary dependencies for your project:

requirements.txt
Copy code
numpy
tensorflow
matplotlib
tk