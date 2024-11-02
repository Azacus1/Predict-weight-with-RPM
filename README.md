Weight Prediction Model Using Neural Networks
This project demonstrates a neural network model for predicting the weight of an item based on various parameters such as RPM, Torque, Power, Wheel Radius, Incline Angle, and Friction Coefficient. The model is trained on synthetically generated data to simulate a relationship between these features and weight.

Table of Contents
Data Preparation
Data Splitting and Scaling
Model Definition
Model Compilation and Training
Evaluation and Prediction
Notes
Data Preparation
Feature Generation: Synthetic data is generated for the following features:

RPM
Torque
Power
Wheel Radius
Incline Angle
Friction Coefficient
Target Variable: The target variable, weight, is synthetically calculated based on a hypothetical relationship with these features.

Data Splitting and Scaling
Data Split: The dataset is divided into training and testing sets with an 80-20 split.
Feature Scaling: All features are standardized to ensure a consistent scale, improving the stability and performance of the neural network model during training.
Model Definition
Network Architecture: A neural network model with three hidden layers is defined, with each hidden layer using the ReLU activation function, ideal for regression tasks.
Output Layer: The output layer consists of a single neuron to predict the continuous weight value.
Model Compilation and Training
Compilation: The model is compiled using the Adam optimizer, with Mean Squared Error as the loss function and Mean Absolute Error (MAE) as the performance metric.
Training: The model is trained over 100 epochs.
Evaluation and Prediction
Model Evaluation: After training, the model is evaluated on the test set, with Mean Absolute Error (MAE) reported as the performance metric.
Prediction Output: Sample predictions are displayed, comparing predicted weight values against the actual weight values to demonstrate model accuracy.
Notes
Hyperparameter Tuning: Model accuracy can be improved by tuning hyperparameters such as the number of layers, neurons, learning rate, and the number of epochs.
Feature Importance: Additional features like RPM decay or other parameters that might impact weight prediction can further improve model performance.
This code serves as a framework for predicting weight based on RPM and related parameters using a neural network model. Contributions and improvements are welcome!
