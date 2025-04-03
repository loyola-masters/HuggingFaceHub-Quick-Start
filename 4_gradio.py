# Import the necessary libraries
import gradio as gr  # Gradio is a library to quickly build and share demos for ML models
import joblib        # joblib is used here to load the trained model from a file
import numpy as np   # NumPy for numerical operations (if needed for array manipulation)

# Load the pre-trained Decision Tree classifier from the joblib file
pipeline = joblib.load("./models/iris_dt.joblib")

# Define a function that takes the four iris measurements as input
# and returns the predicted iris species label.
def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
    # Convert the input parameters into a 2D list/array because
    # scikit-learn's predict() expects a 2D array of shape (n_samples, n_features)
    input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = pipeline.predict(input)
    
    # Convert the prediction to the string label
    if prediction == 0:
        return 'iris-setosa'
    elif prediction == 1:
        return 'Iris-versicolor'
    elif prediction == 2:
        return 'Iris-virginica'
    else:
        return "Invalid prediction"

# Create a Gradio Interface:
# - fn: the function to call for inference
# - inputs: a list of component types to collect user input (in this case, four numeric values)
# - outputs: how the prediction is displayed (in this case, as text)
# - live: whether to update the output in real-time as the user types
interface = gr.Interface(
    fn=predict_iris,
    inputs=["number", "number", "number", "number"],
    outputs="text",
    live=True,
    title="Iris Species Identifier",
    description="Enter the four measurements to predict the Iris species."
)

# Run the interface when this script is executed directly.
# This will launch a local Gradio server and open a user interface in the browser.
if __name__ == "__main__":
    # To create a public link, set the parameter share=True
    interface.launch()
    
'''
# The Flag button allows users (or testers) to mark or “flag”
# a particular input-output interaction for later review.
# When someone clicks Flag, Gradio saves the input values (and often the output) to a log.csv file
# letting you keep track of interesting or potentially problematic cases for debugging or analysis later on
'''
