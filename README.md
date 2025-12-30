Author: Asad Ullah Khan

Project Overview

A web-based machine learning application built with Flask that provides predictive modeling capabilities 
through an intuitive web interface. The application handles data preprocessing, model training, and 
real-time predictions.

Required Python Packages to install first:

Flask>=3.0.0
pandas>=2.0.0
scikit-learn==1.3.0
numpy==1.24.0
scipy==1.9.3
joblib>=1.3.0

Steps Further more:

Before running the app, first extract the models.rar file inside the models folder to get 
the model files, then activate your virtual environment with venv\Scripts\activate, install
required packages using pip install Flask pandas scikit-learn==1.3.0 numpy==1.24.0 scipy==1.9.3 joblib, 
and finally run the application with python app1.py—ensure all file paths in app1.py correctly point to 
your extracted models like model_rf.pkl and the housing.csv dataset.

Project Structure Preview:

Your project at C:\code1 contains app1.py (main Flask app), housing.csv (dataset), a models folder with both .pkl and .joblib model files plus a models.rar archive, a static folder with styles.css,
and a templates folder with index.html.
    
