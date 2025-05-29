import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# Dosyalar
tfidf_dir = '../../ciktilar/tfidf'
input_lema = os.path.join(tfidf_dir, 'tfidf_lemmatized.csv')
input_stem = os.path.join(tfidf_dir, 'tfidf_stemmed.csv')

# Hedef cümle indexi (örnek: 3. açıklama)
target_index = 2

# Lemmatized benzerlik
df_lema = pd.read_csv(input_lema)
cosine_matrix_lema = cosine_similarity(df_lema)
scores_lema = cosine_matrix_lema[target_index]
top5_lema = np.argsort(scores_lema)[::-1][1:6]

print("TF-IDF - Lemmatized:")
for i in top5_lema:
    print(f"Index: {i}, Score: {scores_lema[i]:.4f}")

# Stemmed benzerlik
df_stem = pd.read_csv(input_stem)
cosine_matrix_stem = cosine_similarity(df_stem)
scores_stem = cosine_matrix_stem[target_index]
top5_stem = np.argsort(scores_stem)[::-1][1:6]

print("\nTF-IDF - Stemmed:")
for i in top5_stem:
    print(f"Index: {i}, Score: {scores_stem[i]:.4f}")
