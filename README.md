# Titanic Survival Prediction Pipeline

## Overview

This Python script implements a machine learning pipeline to predict passenger survival on the RMS Titanic, based on the classic Kaggle dataset. The pipeline includes stages for Exploratory Data Analysis (EDA), data cleaning, feature engineering, preprocessing, and building/evaluating a classification model.

The script is structured using Python classes for better organization (`TitanicEDA`, `TitanicDataPreprocessor`, `TitanicModelBuilder`) and incorporates logging to track the workflow. It uses libraries like Pandas for data manipulation, Matplotlib/Seaborn for visualization, Scikit-learn for modeling, and Category Encoders for advanced categorical feature handling.

## Features

*   **Modular Design:** Code organized into distinct classes for EDA, preprocessing, and model building.
*   **Exploratory Data Analysis (EDA):**
    *   Univariate analysis of key numerical (Age, Fare) and categorical (Survived, Pclass, Sex, Embarked) features with visualizations.
    *   Bivariate analysis showing relationships between features and the `Survived` target variable.
    *   Correlation analysis among numerical features using a heatmap.
    *   Brief analysis of potential outliers (high-fare passengers).
    *   Generates plots using `matplotlib` and `seaborn` (displayed during execution).
*   **Data Preprocessing & Cleaning:**
    *   Handles missing values: Imputes missing `Age` with the median, drops the `Cabin` column (due to high missingness), and drops rows with missing `Embarked` values (only 2).
    *   **Feature Engineering:**
        *   Extracts passenger `Title` (e.g., Mr., Mrs., Miss.) from the `Name` column using regex.
        *   Creates `FamilySize` and `IsAlone` features from `SibSp` and `Parch`.
        *   Applies a log transformation (`np.log1p`) to the `Fare` feature to handle skewness.
        *   Creates `AgeGroup` bins using `pd.qcut`.
    *   **Encoding:**
        *   Uses `BinaryEncoder` from `category_encoders` for the extracted `Title` feature (efficient for potentially high cardinality).
        *   Maps `Sex` and `Embarked` to numerical representations.
    *   Drops original and intermediate columns (`Name`, `Ticket`, `PassengerId`, `Cabin`, etc.).
*   **Model Training & Evaluation:**
    *   Uses a `scikit-learn` `Pipeline` to chain `StandardScaler` and `LogisticRegression`.
    *   Splits data into training and testing sets.
    *   Trains the Logistic Regression model.
    *   Evaluates the model using Accuracy and a Classification Report (Precision, Recall, F1-score).
    *   Performs 5-fold cross-validation on the entire dataset to assess model robustness and generalization (using accuracy).
*   **Logging:** Provides informative logs about the pipeline's progress and results using Python's `logging` module.

## Requirements

*   Python 3.x
*   Required Python libraries:
    *   `numpy`
    *   `pandas`
    *   `matplotlib`
    *   `seaborn`
    *   `scikit-learn`
    *   `category_encoders`
    *   `ipython` (implicitly used for `display`, might be needed depending on environment)

## Installation

1.  **Clone or download the script:** Ensure you have the `titanic.py` file.
2.  **Install dependencies:** Open your terminal or command prompt and run:
    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn category_encoders ipython
    # or pip3 install numpy pandas matplotlib seaborn scikit-learn category_encoders ipython
    ```

## Input Data

*   The expected input is a CSV file conforming to the structure of the official `train.csv`.

## Usage

1.  **Modify the data path:** Open `titanic.py` and change the path in the `pd.read_csv()` call within the `main()` function to your local path for `train.csv`.
2.  **Navigate to the directory** containing the script in your terminal.
3.  **Run the script:**
    ```bash
    python titanic.py
    # or python3 titanic.py
    ```
4.  The script will execute the entire pipeline. Logs will be printed to the console, and plots generated by the EDA phase will be displayed sequentially during runtime.

## Outputs

*   **Console Output:** Detailed logs showing the progress of data loading, EDA steps, preprocessing actions, model training results (Accuracy, Classification Report), and cross-validation scores.
*   **Plots:** Visualizations generated during the EDA phase (histograms, count plots, box plots, heatmap) will be displayed on screen during execution. *Note: Plots are not saved to files by default.*

## Code Structure

*   **`TitanicEDA` Class:** Encapsulates all exploratory data analysis functions and plotting.
*   **`TitanicDataPreprocessor` Class:** Contains methods for cleaning data, handling missing values, engineering new features, and encoding categorical variables.
*   **`TitanicModelBuilder` Class:** Responsible for setting up the Scikit-learn pipeline, splitting data, training the model, and evaluating its performance.
*   **`main()` Function:** Orchestrates the overall workflow by instantiating the classes and calling their methods in sequence.
*   **Logging:** Configured at the beginning to provide runtime information.

## License

MIT License

## Author

Andrew Obwocha

