from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.responses import JSONResponse
import numpy as np
import joblib

# Load weights and biases
model = joblib.load('mnist_openheart.joblib')
W1 = model['W1']
W2 = model['W2']
W3 = model['W3']
b1 = model['b1']
b2 = model['b2']
b3 = model['b3']
# Debugging: Check if weights are loaded correctly
print(type(W1), type(b1), type(W2), type(b2), type(W3), type(b3))

# Activation and forward functions
def softplus(x):
    return np.log(1 + np.exp(x))

def mish(x):
    return x * np.tanh(softplus(x))

def mReLU(Z):
    return np.maximum(mish(Z), 0)

def softmax(Z):
    eZ = np.exp(Z - np.max(Z, axis=0))
    return eZ / np.sum(eZ, axis=0)

# Forward propagation
def forward_prop(W1, b1, W2, b2, W3, b3, X):
    Z1 = W1.dot(X) + b1
    A1 = mReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = mReLU(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

# Prediction function
def get_predictions(A3):
    return np.argmax(A3, axis=0)

def make_predictions(X, W1, b1, W2, b2, W3, b3):
    _, _, _, _, _, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
    predictions = get_predictions(A3)
    return predictions

# FastAPI app setup
app = FastAPI()

# Define a model for receiving image data as an array of floats
class ImageData(BaseModel):
    image: list[float]  # Ensure this accepts a list of floats (784 pixel values)

@app.get("/")
async def read_root():
    return {"message": "Welcome to the MNIST API!"}

@app.post('/predict')
async def predict(image_data: ImageData):
    try:
        # Validate that the input image has exactly 784 values
        if len(image_data.image) != 784:
            raise ValueError("Image data should be a list of 784 pixel values.")

        # Convert the input list to a NumPy array and normalize the pixel values
        image_array = np.array(image_data.image).reshape(784, 1) / 255.0  # Reshape to (784, 1)

        # Make predictions using forward propagation
        prediction = make_predictions(image_array, W1, b1, W2, b2, W3, b3)
        
        # Return the prediction as JSON
        return JSONResponse(content={'prediction': int(prediction[0])})
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"An error occurred: {str(e)}")

