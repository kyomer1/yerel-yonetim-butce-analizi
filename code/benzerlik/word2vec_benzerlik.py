import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# Ortak ayarlar
target_index = 2
models_dir = '../../ciktilar/word2vec'

# Cümle ortalamasını al
def average_vector(model, words):
    vectors = [model.wv[w] for w in words if w in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

def calculate_similarity(model, tokenized_data):
    avg_vectors = [average_vector(model, sent) for sent in tokenized_data]
    cosine_sim = cosine_similarity([avg_vectors[target_index]], avg_vectors).flatten()
    return np.argsort(cosine_sim)[::-1][1:6], cosine_sim

# Tokenize et
def tokenize_texts(file_path, col):
    df = pd.read_csv(file_path)
    return df[col].astype(str).apply(lambda x: x.split()).tolist()

# Ana döngü
for set_name, col, path in [
    ('lemmatized', 'aciklama_lemmatized', '../../data/harcamalar_lemmatized.csv'),
    ('stemmed', 'aciklama_stemmed', '../../data/harcamalar_stemmed.csv')
]:
    tokenized = tokenize_texts(path, col)

    for file in os.listdir(models_dir):
        if file.startswith(f'word2vec_{set_name}') and file.endswith('.model'):
            model_path = os.path.join(models_dir, file)
            model = Word2Vec.load(model_path)
            top5, scores = calculate_similarity(model, tokenized)

            print(f"\nModel: {file}")
            for i in top5:
                print(f"Index: {i}, Score: {scores[i]:.4f}")
