# stockify
## Stock Price Prediction Using LSTM Neural Networks
A stock market predictor using advanced machine learning techniques! Focused on building a robust model capable of predicting the future stock prices of any stock, leveraging the power of Long Short-Term Memory (LSTM) neural networks through Keras/TensorFlow.

Included in this repository is the .ipynb file, which is where the ML model was trained. The .keras file attached is the saved Deep Learning model created as a result from the notebook file. I have also included a app.py file which uses Streamlit to effortlessly deploy the application into a webapp.

## Project Overview

We start by gathering historical stock data of the selected ticker from January 1, 2012 (or the latest date), up to the current date. Utilizing the `yfinance` library, we download this extensive dataset, which serves as the foundation for our predictive modeling.

We then move on to exploring the data, particularly focusing on the 'Close' price, which is our target variable. To gain deeper insights and identify potential trends, we calculate and visualize the moving averages over 100 and 200 days.

Recognizing the importance of data preparation, we clean the dataset by removing any null values and split it into training and testing sets, adhering to an 80/20 split. This ensures a robust framework for training our model while retaining a separate dataset for testing its predictive capabilities.

## LSTM Neural Network Architecture

Our model employs a Sequential architecture, intricately designed with multiple LSTM layers interspersed with Dropout layers to mitigate overfitting. This structure allows our network to learn from 100 previous time steps, making it capable of understanding long-term dependencies and subtle patterns in stock price movements.

The model is compiled using the Adam optimizer and mean squared error loss function, striking a balance between efficiency and precision. After training over 50 epochs, we observe the convergence of loss, indicating our model's readiness to make predictions.

## Predictions and Analysis

We test our trained model on the unseen test data, scaling it appropriately to match the training conditions. The model's predictions are then inversely scaled back to their original form, allowing us to directly compare them against the actual stock prices.

Our analysis culminates in a visual representation, plotting both predicted and actual prices, showcasing the model's ability to closely follow the stock's price trajectory.

## Future Predictions

Taking a step further, we utilize our model to predict the stock prices for the upcoming week. This forward-looking feature provides valuable insights, potentially aiding investors in making informed decisions.

## Conclusion and Future Work

This project demonstrates the potential of LSTM neural networks in the realm of stock market predictions. I encourage fellow developers and enthusiasts to experiment with different stocks, adjust the model parameters, and explore further enhancements to the prediction accuracy. The possibilities are limitless, and I am excited to see how this project evolves within the community.


---

*Note: This project is for educational and research purposes only and not be the basis of real-time trading decisions.*
