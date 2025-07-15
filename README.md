Music Popularity Prediction
This project aims to predict the popularity of music tracks using various audio features and metadata. A linear regression model is employed to establish a relationship between these features and a track's popularity score.

1. Load Dataset: The Spotify_data.csv file is loaded into a pandas DataFrame.

2. Data Preprocessing:

3. Non-numeric columns are removed.

The continuous Popularity score is binarized.

Class distribution of the binary target variable is checked.

Handle Imbalance with SMOTE:

Synthetic Minority Over-sampling Technique (SMOTE) is applied to the training data to address class imbalance, provided there are sufficient samples (minimum 6) in the minority class. If not, the original data is used.

4. Train-Test Split: The dataset is divided into training (80%) and testing (20%) sets with random_state=42 for reproducibility.

5. Feature Scaling:

StandardScaler is used to normalize the features in both the training and testing sets, ensuring that features contribute equally to the model.

6. Calculate Class Weights:

compute_class_weight from sklearn.utils is used to calculate balanced class weights based on the training set. These weights are crucial for the ANN to give more importance to the minority class during training.

7. Build the ANN Model:

A Sequential Keras model is constructed with:

An input Dense layer with 64 units and 'relu' activation.

A Dropout layer (0.3) for regularization.

Another Dense layer with 32 units and 'relu' activation.

Another Dropout layer (0.3).

An output Dense layer with 1 unit and 'sigmoid' activation for binary classification.

The model is compiled with binary_crossentropy loss, 'adam' optimizer, and 'Precision', 'Recall', and 'accuracy' as metrics.

8. Train the Model:

The ANN is trained for 100 epochs with a batch size of 16.

A validation split of 20% is used, and the calculated class_weights_dict is passed to the fit method.

9. Evaluate the Model:

The trained model predicts probabilities on the scaled test set.

Predictions are converted to binary classes (0 or 1) using a threshold of 0.5.

The ANN Binary Test Accuracy and a detailed Classification Report (including precision, recall, f1-score, and support for each class) are printed.

10. Plot Training History:

A plot showing the training and validation accuracy over epochs is generated, allowing for visual assessment of model convergence and potential overfitting.

