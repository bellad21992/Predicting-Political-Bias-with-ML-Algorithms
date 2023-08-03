import joblib
from urllib.request import urlopen
from bs4 import BeautifulSoup
import re
import contractions
import nltk
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from urllib.error import HTTPError
from urllib.error import URLError
from http.client import IncompleteRead
from flask import Flask, render_template, request

model = joblib.load('model.pkl')
tfidf_vectorizer = joblib.load('tf.pkl')

def get_title_and_body(URL):
    html = urlopen(URL).read().decode("utf-8")
    htmlParse = BeautifulSoup(html, 'html.parser')
    title = htmlParse.find("title").get_text()
    body = ""
    for p in htmlParse.find_all("p"):
        text = p.get_text()   
        body += text  
    all_text = title + body
    return [all_text, title]

def get_text(URL):
    try:
        get_txt = get_title_and_body(URL)
        text = get_txt[0]
        title = get_txt[1]
    except HTTPError as err:
        text = None
    except URLError as err:
        text = None
    except IncompleteRead as err:
        text = None
    except AttributeError as err:
        if str(err) != "'NoneType' object has no attribute 'get_text'":
            get_txt = get_title_and_body(URL)
            text = get_txt[0]
            title = get_txt[1]
        else:
            text = None
    if text:
        return [text,title]
    return [None, None]
    
def fix_contractions(text):
    fixed_text = []
    for word in text.split():
        fixed_text.append(contractions.fix(word))
    return " ".join(fixed_text)
#spacy.cli.download('en_core_web_sm')
def lemmatize(words):
    nlp=spacy.load('en_core_web_sm',disable=['parser', 'ner'])
    #words = ' '.join([w for w in words])
    text = nlp(words)
    fixed_text = ' '.join([w.lemma_ for w in text])
    return fixed_text
#nltk.download('all', halt_on_error=False)
def remove_stopwords(words):
    stopword_list=nltk.corpus.stopwords.words('english')
    stopword_list.remove('no')
    stopword_list.remove('not')
    stopword_list.remove('nor')
    stopword_list.remove('against')
    stopword_list.remove('now')
    fixed_text = []
    for w in words.split():
        if w not in stopword_list:
            fixed_text.append(w)
    return " ".join(fixed_text)
def clean(text):
    text=text.replace('\xa0'," ")
    text = re.sub(r"(?!(?<=[a-z])'[a-z])[^\w\s]", '', text)
    text=text.lower()
    text=" ".join(text.strip().split())
    text=fix_contractions(text)
    text=remove_stopwords(text)
    text=lemmatize(text)
    return text
    
def show_features(vec, vec_test_features):
    vec_feature_names = vec.get_feature_names_out()
    feature_count = vec_test_features.toarray().sum(axis = 0)
    a = dict(zip(vec_feature_names, feature_count))
    top = []
    counter = Counter(a)
    for word, count in counter.most_common(15):
        top.append(word)
    return top
        
def predict_leaning(url):
    # Step 1: Preprocess the URL and extract the article content
    txt = get_text(url)
    article_content = txt[0]
    title = txt[1]
    scrape = txt[0]
    
    if article_content is not None:
        article_content = clean(article_content)
        cleaned = article_content
        # Step 2: Convert the preprocessed text to features using the TF-IDF Vectorizer
        tfidf_features = tfidf_vectorizer.transform([article_content])
        features = show_features(tfidf_vectorizer,tfidf_features)
        # Step 3: Use the trained model for prediction
        political_leaning = model.predict(tfidf_features)[0]
        if political_leaning == "C":
        	political_leaning = "Center"
        elif political_leaning == "L":
        	political_leaning = "Left"
        else:
        	political_leaning="Right"
        # Step 4: Output the political leaning
        return [title, features, political_leaning, cleaned, scrape]
    else:
        return [None, None, None,None,None]
        
def output(user_input_url):
    predicted_leaning = predict_leaning(user_input_url)
    if predicted_leaning[0]:
        return (predicted_leaning)
    else:
        return ("Error: Unable to access the text from the given URL. ",
              "Possible reasons include subscription costs and pop-up windows.")

app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def index_post():
    bias_result=None
    if request.method=='POST':
        url = request.form['url']
        bias_result = predict_leaning(url)
        return render_template('index.html',bias_result=bias_result)
    return render_template('index.html',bias_result=[None,None,"New"])

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)