import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# 🔢 Örnek simülasyon: her modelin ilk 5 sonucunu örnek olarak tutuyoruz
# Gerçekte bu veriler tfidf_benzerlik.py ve word2vec_benzerlik.py çıktılarından alınacak

model_names = [
    'tfidf_lemmatized', 'tfidf_stemmed',
    'w2v_lema_cbow_w2_d100', 'w2v_lema_skip_w2_d100',
    'w2v_lema_cbow_w4_d100', 'w2v_lema_skip_w4_d100',
    'w2v_lema_cbow_w2_d300', 'w2v_lema_skip_w2_d300',
    'w2v_lema_cbow_w4_d300', 'w2v_lema_skip_w4_d300',
    'w2v_stem_cbow_w2_d100', 'w2v_stem_skip_w2_d100',
    'w2v_stem_cbow_w4_d100', 'w2v_stem_skip_w4_d100',
    'w2v_stem_cbow_w2_d300', 'w2v_stem_skip_w2_d300',
    'w2v_stem_cbow_w4_d300', 'w2v_stem_skip_w4_d300'
]

# 🔁 Her modelin 5 cümle indexini manuel girin (örnek amaçlı sabit veriyorum)
# Gerçekte bu listeler yukarıdaki benzerlik scriptlerinden alınır
model_top5 = {
    name: set(np.random.choice(15, 5, replace=False)) for name in model_names
}

# Jaccard hesapla
def jaccard(set1, set2):
    return len(set1 & set2) / len(set1 | set2)

jaccard_matrix = np.zeros((18, 18))
for i, m1 in enumerate(model_names):
    for j, m2 in enumerate(model_names):
        jaccard_matrix[i][j] = jaccard(model_top5[m1], model_top5[m2])

# Görselleştir
plt.figure(figsize=(12, 10))
sns.heatmap(jaccard_matrix, annot=True, xticklabels=model_names, yticklabels=model_names, fmt=".2f", cmap="Blues")
plt.title("Model Sonuçları Arası Jaccard Benzerlik Matrisi")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()

# Kaydet
os.makedirs("../../ciktilar/jaccard", exist_ok=True)
plt.savefig("../../ciktilar/jaccard/jaccard_matrisi.png")
print("Jaccard matrisi oluşturuldu ve kaydedildi.")
