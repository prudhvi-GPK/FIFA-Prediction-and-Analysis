# FIFA PLAYER PERFORMANCE ANALYSIS AND GOAL PREDICTION

## Project Overview
This project dives into football statistics with a special focus on goalscoring. It aims to analyze player performance and predict whether a given situation during a football match will result in a goal. The dataset used for this project is from Kaggle, containing detailed football event data.

The project is divided into three main phases:
1. **Data Analysis**: Cleaning and exploring the dataset to extract insights related to goalscoring.
2. **Model Building**: Using machine learning techniques to build a predictive model for goal outcomes.
3. **User Interface Development**: Creating a user-friendly web interface where users can input their own data and test the predictive model.

## Key Features:
- Data cleaning and exploratory data analysis (EDA) of football event data.
- Machine learning models including Gradient Boosting Classifier, Logistic Regression, and Soft Voting for prediction.
- Web application built with Flask, providing an interface for users to interact with the data and the prediction model.

## Technologies Used:
- **Python**: Data cleaning, analysis, and machine learning.
- **Flask**: Backend framework to build the web application.
- **HTML/CSS**: Frontend for user interaction.
- **Machine Learning**: Classification and regression trees.

## How to Run:
1. Clone the repository.
2. Install required dependencies from `requirements.txt`.
3. Run the Flask application using `python app.py`.
4. Access the web interface through your browser at `http://localhost:5000`.

## Dataset:
The dataset is sourced from Kaggle and includes detailed football event data such as:
- Time
- Player position
- Assist method
- Body part used to score, etc.

## Predictive Model:
The model predicts the probability of a goal based on the input data, using factors such as the time of the match, assist method, and the player’s position.
