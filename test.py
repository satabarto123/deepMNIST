import pandas as pd
import json

# Load your MNIST dataset (modify the file path as per your setup)
mnist_df = pd.read_csv('mnist_test.csv')  # Replace with the actual path

# Extract the row (e.g., the 5th row)
row = mnist_df.iloc[652].tolist()  # Exclude the label column if needed

# Create a dictionary to store the data in the JSON format
data = {
    "data": row
}

# Save the dictionary as a JSON file
with open('mnist_row.json', 'w') as json_file:
    json.dump(data, json_file)

print("JSON file saved successfully!")
