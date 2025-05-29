import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

df = pd.read_csv('data/harcamalar.csv')
text_column = df.columns[0]

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('turkish'))

def lemmatize_text(text):
    try:
        words = nltk.word_tokenize(str(text).lower())
        words = [w for w in words if w.isalpha() and w not in stop_words]
        return " ".join([lemmatizer.lemmatize(w) for w in words])
    except:
        return ""

df["lemmatized"] = df[text_column].apply(lemmatize_text)
df[["lemmatized"]].to_csv("data/harcamalar_lemmatized.csv", index=False)
