#!/usr/bin/env python
# coding: utf-8
import os
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN --- #
MODEL = "overlap"
if MODEL == "no_overlap":
    RESULTS_FOLDER = "Results/F1_NO_OVER"
else:
    RESULTS_FOLDER = "Results/F1_OVER" 

EXPERIMENTATION_TYPE = "None"
MODE = "All"
RANGE = ""
OUTPUT_FOLDER = "Slope"

# Carpeta de output --> PASAR AL DRIVE
#path = os.path.join(OUTPUT_FOLDER, EXPERIMENTATION_TYPE, MODE, MODEL)
path = os.path.join("Graficos Resultados")
#path = os.path.join("ORIGINAL_RESULTS")
os.makedirs(path, exist_ok=True)
print(f"Resultados se guardarán en: {path}")

Y_TRUE_FILE = os.path.join(RESULTS_FOLDER, "y.csv")

# --- LECTURA DE DATOS ---
y_true_df = pd.read_csv(Y_TRUE_FILE, index_col=0)
y_true = y_true_df.values.squeeze()

all_preds = pd.DataFrame(index=y_true_df.index)

# --- PROCESAR FOLDS ---
for fold_name in os.listdir(RESULTS_FOLDER):
    fold_path = os.path.join(RESULTS_FOLDER, fold_name)
    val_probas_file = os.path.join(fold_path, "val_probas.csv")
    if os.path.exists(val_probas_file):
        fold_probs = pd.read_csv(val_probas_file, index_col=0)
        fold_preds = fold_probs.idxmax(axis=1)
        all_preds = all_preds.join(fold_preds.rename(fold_name), how='left')

# Tomar primera predicción no nula
final_preds = all_preds.bfill(axis=1).iloc[:, 0].values

# --- MÉTRICAS ---
labels = np.unique(y_true)
f1_per_class = f1_score(y_true, final_preds, average=None, labels=labels)
f1_macro = f1_score(y_true, final_preds, average='macro')

# Guardar CSV con métricas en la carpeta de salida
metrics_df = pd.DataFrame({
    'Clase': labels,
    'F1_score': f1_per_class
})
metrics_df.loc[len(metrics_df)] = ['Macro', f1_macro]
metrics_path = os.path.join(path, f"f1_scores_{MODEL}.csv")
metrics_df.to_csv(metrics_path, index=False)

print("F1 por clase:")
for lbl, f1_val in zip(labels, f1_per_class):
    print(f"Clase {lbl}: {f1_val:.4f}")
print(f"F1 macro total: {f1_macro:.4f}")
print(f"Métricas guardadas en: {metrics_path}")

# --- MATRIZ DE CONFUSIÓN ---
cm = confusion_matrix(y_true, final_preds, labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

plt.figure(figsize=(8,6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.title("Matriz de Confusión")
plt.ylabel("Clase Verdadera")
plt.xlabel("Clase Predicha")

# Guardar matriz de confusión
cm_plot_path = os.path.join(path, f"confusion_matrix_{MODEL}.png")
plt.savefig(cm_plot_path, bbox_inches='tight')
plt.close()
print(f"Matriz de confusión guardada en: {cm_plot_path}")

# --- GRÁFICO DE F1 POR CLASE ---
plt.figure(figsize=(8,5))
sns.barplot(x=labels, y=f1_per_class, palette='Blues_d')
plt.title("F1-Score por Clase")
plt.ylabel("F1 Score")
plt.xlabel("Clase")
plt.ylim(0, 1)
for i, val in enumerate(f1_per_class):
    plt.text(i, val + 0.02, f"{val:.2f}", ha='center')

# Guardar gráfico de F1 por clase
f1_plot_path = os.path.join(path, f"f1_per_class_{MODEL}.png")
plt.savefig(f1_plot_path, bbox_inches='tight')
plt.close()
print(f"Gráfico de F1 por clase guardado en: {f1_plot_path}")
