import os

# Asegúrate de que existe la carpeta "data"
os.makedirs("data", exist_ok=True)

# Leer archivo línea por línea
with open("ner-es.train (1).csv", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Crear diccionario con etiquetas conocidas
tag_dict = {}
for line in lines:
    line = line.strip()
    if not line or line == "-":
        continue
    parts = line.split()
    if len(parts) == 2:
        token, tag = parts
        if tag != "O" and tag != "-":
            tag_dict[token] = tag

# Aplicar propagación
new_lines = []
for line in lines:
    stripped = line.strip()
    if not stripped:
        new_lines.append("\n")
        continue
    parts = stripped.split()
    if len(parts) == 2:
        token, tag = parts
        if tag == "-" and token in tag_dict:
            tag = tag_dict[token]
        new_lines.append(f"{token} {tag}\n")
    elif len(parts) == 1:
        token = parts[0]
        tag = tag_dict.get(token, "O")
        new_lines.append(f"{token} {tag}\n")
    else:
        new_lines.append(line)

# Guardar resultado
with open("ner-es.train.labelprop.csv", "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print(" Archivo generado: ner-es.train.labelprop.csv")
