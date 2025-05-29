import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

df = pd.read_csv('data/harcamalar.csv')
text_column = df.columns[0]

stemmer = PorterStemmer()
stop_words = set(stopwords.words('turkish'))

def stem_text(text):
    try:
        words = nltk.word_tokenize(str(text).lower())
        words = [w for w in words if w.isalpha() and w not in stop_words]
        return " ".join([stemmer.stem(w) for w in words])
    except:
        return ""

df["stemmed"] = df[text_column].apply(stem_text)
df[["stemmed"]].to_csv("data/harcamalar_stemmed.csv", index=False)
