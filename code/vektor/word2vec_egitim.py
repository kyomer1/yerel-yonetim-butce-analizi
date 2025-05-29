import pandas as pd
from gensim.models import Word2Vec
import os

# Dosya yolları
lemmatized_file = '../../data/harcamalar_lemmatized.csv'
stemmed_file = '../../data/harcamalar_stemmed.csv'
output_dir = '../../ciktilar/word2vec'
os.makedirs(output_dir, exist_ok=True)

# Verileri hazırla
def tokenize_column(df, column):
    return df[column].astype(str).apply(lambda x: x.split()).tolist()

df_lema = pd.read_csv(lemmatized_file)
df_stem = pd.read_csv(stemmed_file)

lemmatized_tokens = tokenize_column(df_lema, 'aciklama_lemmatized')
stemmed_tokens = tokenize_column(df_stem, 'aciklama_stemmed')

# Parametreler
params = [
    {'model_type': 'cbow', 'window': 2, 'vector_size': 100},
    {'model_type': 'skipgram', 'window': 2, 'vector_size': 100},
    {'model_type': 'cbow', 'window': 4, 'vector_size': 100},
    {'model_type': 'skipgram', 'window': 4, 'vector_size': 100},
    {'model_type': 'cbow', 'window': 2, 'vector_size': 300},
    {'model_type': 'skipgram', 'window': 2, 'vector_size': 300},
    {'model_type': 'cbow', 'window': 4, 'vector_size': 300},
    {'model_type': 'skipgram', 'window': 4, 'vector_size': 300},
]

# Model eğitimi
for param in params:
    sg = 1 if param['model_type'] == 'skipgram' else 0
    model_lema = Word2Vec(
        sentences=lemmatized_tokens,
        vector_size=param['vector_size'],
        window=param['window'],
        sg=sg,
        min_count=1,
        epochs=100
    )
    model_stem = Word2Vec(
        sentences=stemmed_tokens,
        vector_size=param['vector_size'],
        window=param['window'],
        sg=sg,
        min_count=1,
        epochs=100
    )

    # Model isimlendirme
    name = f"{param['model_type']}_win{param['window']}_dim{param['vector_size']}"
    model_lema.save(os.path.join(output_dir, f"word2vec_lemmatized_{name}.model"))
    model_stem.save(os.path.join(output_dir, f"word2vec_stemmed_{name}.model"))
    print(f"{name} modelleri kaydedildi.")
