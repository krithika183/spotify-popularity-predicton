Project: Song Popularity Predictor

Description:
This project provides a web application to predict the popularity of a song based on its audio features and release information. It uses a Linear Regression model trained on Spotify data.

Files Included:
- app.py: The Flask backend application that serves the web page, loads the machine learning model, and handles prediction requests.
- templates/index.html: The frontend HTML file that provides the user interface for inputting song features and displaying predictions.
- popularity_predictor_final.pkl: The pre-trained machine learning model (generated from the Jupyter notebook) used for making predictions.
- Spotify_data.csv: The dataset used for training the model and for median imputation of missing features in the backend.

Setup Instructions:

1.  **Project Structure:**
    Create a new directory for your project (e.g., `song_popularity_app`).
    Inside this directory, create a subfolder named `templates`.

    Your project structure should look like this:
    ```
    song_popularity_app/
    ├── app.py
    ├── popularity_predictor_final.pkl
    ├── Spotify_data.csv
    └── templates/
        └── index.html
    ```

2.  **Save the Files:**
    -   Save the `app.py` code (provided previously) as `app.py` in the `song_popularity_app/` directory.
    -   Save the `index.html` code (provided previously) as `index.html` inside the `templates/` subfolder.
    -   Ensure your trained model file `popularity_predictor_final.pkl` (which you generated from the Jupyter notebook) is placed directly in the `song_popularity_app/` directory.
    -   Ensure your `Spotify_data.csv` file is also placed directly in the `song_popularity_app/` directory. This file is used by `app.py` for calculating feature medians for imputation if a user doesn't provide all inputs.

3.  **Install Dependencies:**
    Open your terminal or command prompt. Navigate to your project directory (`song_popularity_app/`).
    Run the following command to install the required Python libraries:
    ```bash
    pip install Flask scikit-learn pandas joblib
    ```

How to Run the Application:

1.  **Navigate to Project Directory:**
    In your terminal or command prompt, make sure you are in the `song_popularity_app/` directory.

2.  **Run the Flask Application:**
    Execute the following command:
    ```bash
    python app.py
    ```
    You should see output indicating that the Flask development server is running, typically on `http://127.0.0.1:5000/`.

How to Access the Application:

1.  **Open in Browser:**
    Open your web browser and go to the address displayed in your terminal, usually:
    ```
    [http://127.0.0.1:5000/](http://127.0.0.1:5000/)
    ```
    You will see the "Song Popularity Predictor" interface.

Notes/Troubleshooting:

-   **`Spotify_data.csv`:** The `app.py` script loads `Spotify_data.csv` to calculate medians for feature imputation. If this file is missing or corrupted, the application might encounter errors during startup or prediction.
-   **Model File:** Ensure `popularity_predictor_final.pkl` is correctly placed and not corrupted. If the model fails to load, predictions will not work.
-   **Debug Mode:** `app.py` runs in debug mode (`debug=True`) by default, which is useful for development as it provides detailed error messages and reloads automatically on code changes. For a production environment, you should change `debug=True` to `debug=False` in `app.py`.
-   **Dependencies:** If you encounter `ModuleNotFoundError` or similar, double-check that all dependencies listed in "Install Dependencies" step are correctly installed.