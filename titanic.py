
"""
Titanic Survival Prediction Model
--------------------------------
This script performs exploratory data analysis and builds a prediction model
for the Titanic dataset. It includes data cleaning, feature engineering,
and a logistic regression model to predict passenger survival.

Author: [Your Name]
Date: January 15, 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from category_encoders import BinaryEncoder
import re
from typing import List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TitanicEDA:
    """
    Handles Exploratory Data Analysis for the Titanic dataset.
    """
    def __init__(self, df: pd.DataFrame):
        """Initialize with dataframe."""
        self.df = df.copy()
        
    def plot_univariate_analysis(self) -> None:
        """Perform univariate analysis for key features."""
        logger.info("Performing univariate analysis...")
        
        # Numeric features distributions
        numeric_features = ['Age', 'Fare']
        for feature in numeric_features:
            plt.figure(figsize=(8, 6))
            sns.histplot(self.df[feature], bins=20, kde=True)
            plt.title(f'Distribution of {feature}')
            plt.show()
            
            # Log basic statistics
            logger.info(f"\n{feature} statistics:")
            logger.info(f"Mean: {self.df[feature].mean():.2f}")
            logger.info(f"Median: {self.df[feature].median():.2f}")
            logger.info(f"Std: {self.df[feature].std():.2f}")
        
        # Categorical features distributions
        categorical_features = ['Survived', 'Pclass', 'Sex', 'Embarked']
        for feature in categorical_features:
            plt.figure(figsize=(8, 6))
            sns.countplot(x=feature, data=self.df)
            plt.title(f'Distribution of {feature}')
            plt.show()
            
            # Log value counts
            logger.info(f"\n{feature} value counts:")
            logger.info(self.df[feature].value_counts())
    
    def plot_survival_relationships(self) -> None:
        """Analyze relationships between features and survival."""
        logger.info("Analyzing survival relationships...")
        
        # Categorical features vs Survival
        categorical_features = ['Pclass', 'Sex', 'Embarked']
        for feature in categorical_features:
            plt.figure(figsize=(10, 6))
            sns.countplot(x='Survived', hue=feature, data=self.df)
            plt.title(f'Survival Distribution by {feature}')
            plt.show()
            
            # Log survival rates
            survival_rates = self.df.groupby(feature)['Survived'].mean()
            logger.info(f"\nSurvival rates by {feature}:")
            logger.info(survival_rates)
        
        # Numeric features vs Survival
        numeric_features = ['Age', 'Fare']
        for feature in numeric_features:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='Survived', y=feature, data=self.df)
            plt.title(f'Distribution of {feature} by Survival')
            plt.show()
            
            # Log average values by survival
            avg_by_survival = self.df.groupby('Survived')[feature].mean()
            logger.info(f"\nAverage {feature} by survival status:")
            logger.info(avg_by_survival)
    
    def plot_correlation_analysis(self) -> None:
        """Analyze correlations between numeric features."""
        logger.info("Performing correlation analysis...")
        
        # Select numeric columns
        numeric_df = self.df.select_dtypes(include=['int64', 'float64'])
        
        # Calculate and plot correlation matrix
        correlation_matrix = numeric_df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
        plt.title('Correlation Matrix of Numeric Features')
        plt.show()
        
        # Log strong correlations
        logger.info("\nStrong correlations (|r| > 0.3):")
        for i in range(len(correlation_matrix.columns)):
            for j in range(i):
                if abs(correlation_matrix.iloc[i, j]) > 0.3:
                    logger.info(f"{correlation_matrix.columns[i]} vs {correlation_matrix.columns[j]}: {correlation_matrix.iloc[i, j]:.3f}")
    
    def analyze_fare_outliers(self) -> None:
        """Analyze fare outliers."""
        logger.info("Analyzing fare outliers...")
        
        # Display high fare passengers
        high_fare_threshold = self.df['Fare'].quantile(0.95)
        high_fare_passengers = self.df[self.df['Fare'] > high_fare_threshold]
        
        logger.info(f"\nPassengers with fares > {high_fare_threshold:.2f}:")
        display(high_fare_passengers[['Pclass', 'Sex', 'Age', 'Fare', 'Survived']])
    
    def perform_complete_eda(self) -> None:
        """Perform complete exploratory data analysis."""
        logger.info("Starting complete exploratory data analysis...")
        
        # Set plot style
        plt.style.use('seaborn')
        
        # Perform all analyses
        self.plot_univariate_analysis()
        self.plot_survival_relationships()
        self.plot_correlation_analysis()
        self.analyze_fare_outliers()
        
        logger.info("Exploratory data analysis completed.")


class TitanicDataPreprocessor:
    """
    A class to handle all data preprocessing steps for the Titanic dataset.
    """
    def __init__(self, df: pd.DataFrame):
        """Initialize with raw dataframe."""
        self.df = df.copy()
        self.numerical_features = ['Age', 'Fare']
        self.categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title']
        
    def extract_title(self, text: str, titles: Optional[List[str]] = None) -> Optional[str]:
        """
        Extract title from passenger name.
        
        Args:
            text: Passenger name string
            titles: List of possible titles to extract
            
        Returns:
            Extracted title or None if no title found
        """
        if titles is None:
            titles = ['Mr', 'Mrs', 'Ms', 'Miss', 'Dr', 'Prof', 'Rev', 'Sir',
                     'Master', 'Dame', 'Lady', 'Lord', 'Captain', 'Don', 'Mme',
                     'Major', 'Jonkheer', 'Countess', 'Capt', 'Col', 'Mlle']
        
        patterns = [f'{title}\\.?\\s*' for title in titles]
        full_pattern = r'\b(' + '|'.join(patterns) + r')\b'
        
        match = re.search(full_pattern, text, re.IGNORECASE)
        if match:
            title = match.group(1).strip().title()
            return f"{title}." if not title.endswith('.') else title
        return None

    def handle_missing_values(self) -> None:
        """Handle missing values in the dataset."""
        logger.info("Handling missing values...")
        
        # Drop rows with missing Embarked (only 2 in the dataset)
        self.df = self.df.dropna(subset=['Embarked'])
        
        # Fill missing Age with median (more robust than mean)
        self.df['Age'].fillna(self.df['Age'].median(), inplace=True)
        
        # Drop Cabin (too many missing values to be useful)
        self.df.drop('Cabin', axis=1, inplace=True)

    def create_features(self) -> None:
        """Create and transform features."""
        logger.info("Creating and transforming features...")
        
        # Extract titles from names
        self.df['Title'] = self.df['Name'].apply(self.extract_title)
        
        # Create family size feature
        self.df['FamilySize'] = self.df['SibSp'] + self.df['Parch'] + 1
        self.df['IsAlone'] = (self.df['FamilySize'] == 1).astype(int)
        
        # Fare binning (log transform to handle skewness)
        self.df['Fare'] = np.log1p(self.df['Fare'])
        
        # Age binning
        self.df['AgeGroup'] = pd.qcut(self.df['Age'], 5, labels=['Very Young', 'Young', 'Middle', 'Senior', 'Elderly'])

    def encode_categorical(self) -> None:
        """Encode categorical variables."""
        logger.info("Encoding categorical variables...")
        
        # Binary encoding for Title (handles high cardinality better than one-hot)
        encoder = BinaryEncoder(cols=['Title'])
        encoded_titles = encoder.fit_transform(self.df[['Title']])
        self.df = pd.concat([self.df, encoded_titles], axis=1)
        
        # Simple mapping for binary/ordinal categories
        self.df['Sex'] = self.df['Sex'].map({'male': 1, 'female': 0})
        self.df['Embarked'] = self.df['Embarked'].map({'S': 2, 'C': 1, 'Q': 0})

    def clean_data(self) -> pd.DataFrame:
        """
        Execute all preprocessing steps and return cleaned dataframe.
        
        Returns:
            Cleaned and preprocessed DataFrame
        """
        logger.info("Starting data preprocessing...")
        
        self.handle_missing_values()
        self.create_features()
        self.encode_categorical()
        
        # Drop unnecessary columns
        columns_to_drop = ['Name', 'Ticket', 'PassengerId', 'Title', 'Age', 'SibSp', 'Parch']
        self.df.drop(columns_to_drop, axis=1, inplace=True)
        
        logger.info("Data preprocessing completed")
        return self.df

class TitanicModelBuilder:
    """
    Handles model creation, training, and evaluation for Titanic survival prediction.
    """
    def __init__(self, df: pd.DataFrame):
        """Initialize with preprocessed dataframe."""
        self.df = df
        self.X = df.drop('Survived', axis=1)
        self.y = df['Survived']
        self.model = None
        
    def create_pipeline(self) -> Pipeline:
        """Create sklearn pipeline with preprocessing and model."""
        return Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])
    
    def train_evaluate(self, test_size: float = 0.2) -> None:
        """
        Train model and evaluate performance.
        
        Args:
            test_size: Proportion of dataset to use for testing
        """
        logger.info("Starting model training and evaluation...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42
        )
        
        # Create and train pipeline
        self.model = self.create_pipeline()
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Evaluate model
        logger.info(f"Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        logger.info("\nClassification Report:\n" + classification_report(y_test, y_pred))
        
        # Perform cross-validation
        cv_scores = cross_val_score(self.model, self.X, self.y, cv=5)
        logger.info(f"\nCross-validation scores: {cv_scores}")
        logger.info(f"Average CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

def main():
    """Main execution function."""
    # Load data
    try:
        df = pd.read_csv('/kaggle/input/titanic/train.csv')
        logger.info("Data loaded successfully")
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return
    
    # Perform EDA
    eda = TitanicEDA(df)
    eda.perform_complete_eda()
    
    # Preprocess data
    preprocessor = TitanicDataPreprocessor(df)
    cleaned_df = preprocessor.clean_data()
    
    # Train and evaluate model
    model_builder = TitanicModelBuilder(cleaned_df)
    model_builder.train_evaluate()

if __name__ == "__main__":
    main()