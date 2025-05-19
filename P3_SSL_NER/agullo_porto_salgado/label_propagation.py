from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.semi_supervised import LabelPropagation
import numpy as np

# Cargar datos desde archivo
file_path = "data/ner-es.train.csv"
with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Parsear tokens y etiquetas
tokens = []
labels = []
for line in lines:
    line = line.strip()
    if line == "":
        continue
    try:
        token, label = line.split()
    except ValueError:
        continue
    tokens.append(token)
    labels.append(label)

# Mapeo de etiquetas a índices
NUM_LABELED = 268_781
label_names = sorted(set(labels[:NUM_LABELED]))
label_to_index = {label: idx for idx, label in enumerate(label_names)}
index_to_label = {idx: label for label, idx in label_to_index.items()}

# Convertir etiquetas a índices, -1 para no etiquetados
y = []
for i, label in enumerate(labels):
    if i < NUM_LABELED:
        y.append(label_to_index[label])
    else:
        y.append(-1)

# Vectorización de tokens
vectorizer = TfidfVectorizer(lowercase=False, analyzer='char_wb', ngram_range=(2, 3))
X_sparse = vectorizer.fit_transform(tokens)

# ⚠️ Requiere mucha RAM, convertir a denso puede fallar en máquinas sin recursos suficientes
X_dense = X_sparse.toarray()

# Aplicar Label Propagation
lp_model = LabelPropagation()
lp_model.fit(X_dense, y)

# Obtener etiquetas resultantes
predicted_indices = lp_model.transduction_
predicted_labels = [index_to_label.get(idx, "UNKNOWN") for idx in predicted_indices]

# Guardar resultados en un nuevo archivo
output_path = "data/ner-es_filled_lp.csv"
with open(output_path, "w", encoding="utf-8") as f:
    for token, label in zip(tokens, predicted_labels):
        f.write(f"{token} {label}\n")

print(f"Etiquetas propagadas y guardadas en: {output_path}")