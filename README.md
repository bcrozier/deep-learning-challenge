# deep-learning-challenge
Module 21 Challenge
Data Preprocessing:

Target Variable:

The target variable for our model is IS_SUCCESSFUL, which indicates whether the funding was used effectively (1) or not (0).
Features for the Model:

The features for our model include various columns from the dataset, such as APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, INCOME_AMT, and SPECIAL_CONSIDERATIONS.
Variables to be Removed:

The EIN and NAME columns were removed from the input data as they are neither targets nor features for our model.
Compiling, Training, and Evaluating the Model:

Model Architecture:

The model consists of three fully connected (dense) layers: an input layer with 128 neurons and ReLU activation function, one hidden layer with 64 neurons and ReLU activation function, and an output layer with 1 neuron and sigmoid activation function.
ReLU activation function was chosen for its ability to introduce non-linearity, which allows the model to learn complex patterns in the data.
Model Performance:

The target model performance was set at achieving an accuracy higher than 75%.
After training the model for 50 epochs with a batch size of 64, the model achieved a test accuracy of approximately 75.2%, surpassing the target performance.
Steps to Increase Model Performance:

To optimize the model, various approaches were tried, including adjusting the number of neurons and layers, experimenting with different activation functions, and tuning hyperparameters such as the number of epochs and batch size.
Additionally, data preprocessing techniques such as feature scaling and one-hot encoding were applied to improve model convergence and performance.
Summary:

The deep learning model developed for Alphabet Soup achieved the target model performance, with a test accuracy exceeding 75%. By carefully selecting the model architecture, optimizing hyperparameters, and preprocessing the data appropriately, we were able to build a model that effectively predicts the success of funding applicants.

While the current model performed satisfactorily, there is always room for further improvement. One recommendation for solving this classification problem is to explore more complex neural network architectures, such as convolutional neural networks (CNNs) or recurrent neural networks (RNNs). These architectures are well-suited for handling sequential or image data and may capture additional patterns that could enhance prediction accuracy.

In summary, while the current model achieved the desired performance, exploring alternative architectures and continuing to refine preprocessing techniques could lead to further improvements in predictive accuracy.

