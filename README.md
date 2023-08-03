# Predicting-Political-Bias-with-ML-Algorithms

The predictions from this webapp are generated using ensemble methods of stacking multiple machine learning models. The supervised learning classifiers used in the final stacking classifier include logistic regression, random forest, support vector machine, multinomial naive bayes, AdaBoost, and XGBoost. Features, or important words, are extracted using a TFIDF Vectorizer, and the final stacking classifier is 79% accurate in its predictions of news articles as Left, Center, or Right leaning.


To run the web app:
1. Download 'app.py', 'index.html', 'model.pkl', and 'tf.pkl'. Put all four of these files into the same folder, then put 'index.html' into it's own folder named "Templates". 
2. Run 'app.py' in Terminal by navigating to the folder where the files are located, and typing "python app.py" into the terminal.
3. Copy the link: "http://192.168.0.3:5000/" into your brower, and enter a URL of a political news article into the box to predict its political leaning!


Files in this repository:
1. Data_Cleaning.ipynb
2. Data_Exploration.ipynb
3. ML_Models.ipynb
4. model.pkl
5. tf.pkl
6. app.py
7. index.html
