# Import necessary libraries
import numpy as np                   # For numerical operations
import pandas as pd                  # For data handling
import matplotlib.pyplot as plt      # For plotting
from sklearn.linear_model import LinearRegression  # For building the regression model

# 1. Create the dataset using a dictionary and convert it to a Pandas DataFrame
data = {
    'SquareFootage': [1000, 1500, 2000, 2500, 3000],
    'HousePrice': [200, 250, 300, 350, 400]  # House Price in $1000s
}
df = pd.DataFrame(data)

# 2. Separate the independent variable (X) and the dependent variable (y)
X = df[['SquareFootage']]  # Features
y = df['HousePrice']       # Target variable

# 3. Create and train the Linear Regression model
model = LinearRegression()
model.fit(X, y)

def plot_prediction(user_input, predicted_value):
    """
    This function plots the original data points, the regression line,
    and highlights the user's predicted value on the graph.
    """
    plt.figure(figsize=(8, 6))
    
    # Plot the original data points
    plt.scatter(X, y, color='blue', label='Data Points')
    
    # Plot the regression line using the predictions on original X values
    plt.plot(X, model.predict(X), color='red', label='Regression Line')
    
    # Highlight the user's prediction with a green 'x'
    plt.scatter(user_input, predicted_value, color='green', marker='x', s=100, label='Your Prediction')
    
    # Labeling the graph
    plt.xlabel('Square Footage')
    plt.ylabel('House Price (in $1000s)')
    plt.title('House Price Prediction using Linear Regression')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    try:
        # 4. Prompt the user for input square footage
        user_input = float(input("Enter the square footage: "))
        
        # 5. Predict the house price using the model
        predicted_price = model.predict(np.array([[user_input]]))[0]
        print(f"Predicted house price for {user_input} square feet: {predicted_price:.2f} (in $1000s)")
        
        # 6. Plot the data along with the prediction
        plot_prediction(user_input, predicted_price)
    except ValueError:
        print("Please enter a valid numerical value for square footage.")

# Run the program
if __name__ == '__main__':
    main()
