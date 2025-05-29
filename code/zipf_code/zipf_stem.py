import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import os

input_file = '../../data/harcamalar_stemmed.csv'
output_dir = '../../ciktilar/zipf_grafikleri'
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(input_file)
text = ' '.join(df['aciklama_stemmed'].astype(str).tolist()).lower().split()
freq = Counter(text)
sorted_freq = sorted(freq.values(), reverse=True)

ranks = np.arange(1, len(sorted_freq) + 1)
plt.figure(figsize=(10, 6))
plt.loglog(ranks, sorted_freq)
plt.xlabel("Kelime Sırası (Rank)")
plt.ylabel("Frekans")
plt.title("Zipf Grafiği (Stemmed)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'zipf_stem.png'))
print("Stemmed Zipf grafiği kaydedildi.")
