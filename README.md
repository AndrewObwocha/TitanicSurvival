# Titanic Survival Predictor

A machine learning model that predicts passenger survival on the Titanic.

## Description

This project analyzes the famous Titanic dataset to build a predictive model that determines which passengers would have survived the disaster. It demonstrates data cleaning, feature engineering, and classification techniques.

## Features

- Predicts survival probability based on passenger attributes
- Visualizes survival patterns across different demographics
- Implements various classification algorithms for comparison

## Installation

1. git clone https://github.com/AndrewObwocha/TitanicSurvival.git
2. cd TitanicSurvival
3. pip install -r requirements.txt

## Usage

1. Run the analysis and training: python train.py
2. Make predictions on new data: python predict.py --input passenger_data.csv

## Data

The model uses the Titanic dataset with features including:
- Passenger class
- Sex
- Age
- Number of siblings/spouses aboard
- Number of parents/children aboard
- Ticket fare
- Port of embarkation

## Requirements

- Python 3.0
- pandas
- scikit-learn
- matplotlib
- seaborn

## Results

The current model achieves approximately 80% prediction accuracy on test data.

## License

MIT License

## Author

Andrew Obwocha

