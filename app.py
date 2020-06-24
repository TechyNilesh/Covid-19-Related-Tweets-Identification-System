# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import re
import ftfy


app = Flask(__name__)

# Load the LogisticRegression model and Tfidfvectorizetion object from disk
filename = 'covid-msg-lr-model.pkl'
model = pickle.load(open(filename, 'rb'))
vect = pickle.load(open('tf-vect-transform.pkl', 'rb'))

#text cleaning
def cleanTxt(text):
    text = ' '.join(re.sub("([^0-9A-Za-z \t])", " ", text).split()) #remove punctuation
    text = ftfy.fix_text(text) #fix weirdly encoded texts 
    text = text.lower() # all to lower latter
    #stop words
    #stop_words = set(stopwords.words('english'))
    #word_tokens = nltk.word_tokenize(text) 
    #filtered_sentence = [w for w in word_tokens if not w in stop_words]
    #Word Lemmatization
    #text = WordNetLemmatizer().lemmatize(text,"v")
    #joining text
    text = ' '.join(filtered_sentence)
    return text

#predication Function
def predcovidmsg(text):
    text = cleanTxt(text)
    vtext = vect.transform([text])
    result = model.predict_proba(vtext)
    return result

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        prediction = predcovidmsg(text)
        bar_labels=['Not COVID-19 Related','COVID-19 Related']
        bar_values=[prediction[0][0]*100,prediction[0][1]*100]
        print(bar_values)
        return render_template('index.html',labels=bar_labels, values=bar_values,text=text)

if __name__ == '__main__':
    app.run(debug=True)
