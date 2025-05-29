import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# Girdi dosyaları
input_lema = '../../data/harcamalar_lemmatized.csv'
input_stem = '../../data/harcamalar_stemmed.csv'

# Çıktı klasörü
output_dir = '../../ciktilar/tfidf'
os.makedirs(output_dir, exist_ok=True)

# Lemmatized veri
df_lema = pd.read_csv(input_lema)
vectorizer_lema = TfidfVectorizer()
X_lema = vectorizer_lema.fit_transform(df_lema['aciklama_lemmatized'])

df_tfidf_lema = pd.DataFrame(X_lema.toarray(), columns=vectorizer_lema.get_feature_names_out())
df_tfidf_lema.to_csv(os.path.join(output_dir, 'tfidf_lemmatized.csv'), index=False)
print("TF-IDF lemmatized CSV oluşturuldu.")

# Stemmed veri
df_stem = pd.read_csv(input_stem)
vectorizer_stem = TfidfVectorizer()
X_stem = vectorizer_stem.fit_transform(df_stem['aciklama_stemmed'])

df_tfidf_stem = pd.DataFrame(X_stem.toarray(), columns=vectorizer_stem.get_feature_names_out())
df_tfidf_stem.to_csv(os.path.join(output_dir, 'tfidf_stemmed.csv'), index=False)
print("TF-IDF stemmed CSV oluşturuldu.")
