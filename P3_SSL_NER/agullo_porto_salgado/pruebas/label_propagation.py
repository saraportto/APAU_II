import pandas as pd
import numpy as np
from sklearn.semi_supervised import LabelPropagation
from sklearn.feature_extraction.text import CountVectorizer
import os

# Cargar el archivo línea por línea manteniendo frases separadas por saltos de línea
with open("ner-es.train (1).csv", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Procesar tokens y etiquetas
tokens = []
tags = []

for line in lines:
    stripped = line.strip()
    if not stripped:
        continue
    parts = stripped.split()
    if len(parts) == 2:
        token, tag = parts
        tokens.append(token)
        tags.append(tag)
    elif len(parts) == 1:
        token = parts[0]
        tokens.append(token)
        tags.append("-")  # Si no hay etiqueta, asumimos desconocida

# Convertimos etiquetas a números
unique_tags = sorted(set(t for t in tags if t != "-" and t != "O"))
tag_to_id = {tag: i for i, tag in enumerate(unique_tags)}
id_to_tag = {i: tag for tag, i in tag_to_id.items()}

# Creamos etiquetas numéricas para LabelPropagation (los no etiquetados son -1)
y = []
for tag in tags:
    if tag in tag_to_id:
        y.append(tag_to_id[tag])
    else:
        y.append(-1)

y = np.array(y)

# Vectorizar tokens (simplemente por texto, aunque puedes usar mejores embeddings)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(tokens)

# Aplicar label propagation
lp_model = LabelPropagation(kernel='knn', n_neighbors=20, max_iter=1000)
lp_model.fit(X, y)

# Recuperar etiquetas predichas
y_full = lp_model.transduction_

# Reconstruir etiquetas de texto
tags_final = []
for orig, pred in zip(tags, y_full):
    if orig == "-":
        if pred in id_to_tag:
            tags_final.append(id_to_tag[pred])
        else:
            tags_final.append("O")
    else:
        tags_final.append(orig)

# Guardar resultado
os.makedirs("data", exist_ok=True)
with open("data/ner-es.train.labelprop.csv", "w", encoding="utf-8") as f:
    for token, tag in zip(tokens, tags_final):
        f.write(f"{token} {tag}\n")

print(" Etiquetas propagadas con LabelPropagation y guardadas en data/ner-es.train.labelprop.csv")


