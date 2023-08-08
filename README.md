# Predicting-Political-Bias-with-ML-Algorithms

The political bias classifications from this webapp are generated using ensemble methods of stacking multiple machine learning models, and important words or features are extracted using a TFIDF Vectorizer. The supervised learning classifiers used in the final stacking classifier include logistic regression, random forest, support vector machine, multinomial naive bayes, AdaBoost, and XGBoost. After hypertuning parameters using GridSearchCV, the final stacking classifier is 79% accurate in its predictions of news articles as Left, Center, or Right leaning.


To run the web app:
1. Download 'app.py', 'index.html', 'model.pkl', and 'tf.pkl'. Put all four of these files into the same folder, then put 'index.html' into it's own folder named "Templates". 
2. Make sure you have Python 3 installed on your computer. Next, install libaries/packages by typing the following commands into Terminal:
* pip3 install joblib==1.2.0
* pip3 install bs4==4.12.2
* pip3 install contractions==0.1.73
* pip3 install nltk==3.8.1
* pip3 install spacy==3.5.3,
* pip3 install scikit-learn==1.2.2
* pip3 install flask==2.2.3
* python -m spacy download en_core_web_sm
* python -m nltk.downloader all
3. In the Terminal, navigate to the folder where the four files are located by typing "cd _____" with the path to your directory containing the four files in the blank.
4. Type "python app.py" into the terminal to run the app.
5. Copy the second link returned from the output: "http://192.168.0.3:5000/" into your brower, and enter a URL of a political news article into the box to predict its political leaning!


Files in this repository:
1. Data_Cleaning.ipynb
2. Data_Exploration.ipynb
3. ML_Models.ipynb
4. model.pkl
5. tf.pkl
6. app.py
7. index.html
