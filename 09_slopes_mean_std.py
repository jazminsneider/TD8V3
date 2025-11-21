import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import RANSACRegressor, LinearRegression
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

tipos = ["overlap", "no_overlap"]
for tipo in tipos:
    if tipo == "overlap":
        X = pd.read_csv("X_Y/overlap/dev/X.csv", index_col=0)
        S = pd.read_csv("X_Y/overlap/dev/sessions.csv", index_col=0)
        Y = pd.read_csv("X_Y/overlap/dev/Y.csv", index_col=0)
    else:
        X = pd.read_csv("X_Y/no_overlap/dev/X.csv", index_col=0)
        S = pd.read_csv("X_Y/no_overlap/dev/sessions.csv", index_col=0)
        Y = pd.read_csv("X_Y/no_overlap/dev/Y.csv", index_col=0)


    print(f"✅ Dataset leído con {X.shape[0]} filas y {X.shape[1]} columnas.")
    print(f"✅ Sessions leído con {S.shape[0]} filas.") 


    # Parámetros
    cantidades_de_ms = [25, 50, 75, 99]
    regresiones_dict = {c: [] for c in cantidades_de_ms}
    slopes_dict = {c: [] for c in cantidades_de_ms}
    y = []

    slopes_dict_intensity = {c: [] for c in cantidades_de_ms}
    regresiones_dict_intensity = {c: [] for c in cantidades_de_ms}


    # Aca guardamos los means y stds de PITCH
    pitch_means = []
    pitch_stds = []

    # Aca guardamos los means y stds de JITTER
    jitter_means = []
    jitter_stds = []

    # Aca guardamos los means y stds de SHIMMER
    shimmer_means = []
    shimmer_stds = []

    # Aca guardamos los means y stds de LOGHNR
    log_means = []
    log_stds = []

    # Aca guardamos los means y stds de intensity
    intensity_means = []
    intensity_stds = []


    for idx, row in tqdm(X.iterrows(), total=X.shape[0]):
        canal = row["speaker_1_channel"]
        canal_str = f"pitch_standardized_{canal}"

        # Calcular promedio y std del pitch (solo canal activo)

        cols_range = [f"{i}-{canal_str}" for i in range(100) if f"{i}-{canal_str}" in X.columns]
        #print(cols_range)

        if not cols_range:
            pitch_means.append(np.nan)
            pitch_stds.append(np.nan)
            continue

        datos_y = row[cols_range].to_numpy(dtype=float)
    
        datos_y = np.where(datos_y == -15, np.nan, datos_y)

        if np.all(np.isnan(datos_y)):
            #print("Todos los valores son NaN")
            #print(idx)
            #print(Y.loc[idx].values[0])
            y.append(Y.loc[idx].values[0])


        mean_pitch = np.nanmean(datos_y)
        std_pitch = np.nanstd(datos_y)

        pitch_means.append(mean_pitch)
        pitch_stds.append(std_pitch)


        # Calcular promedio y std del jitter (solo canal activo)
        canal_str_jitter = f"jitter_standardized_{canal}"
        cols_jitter = [f"{i}-{canal_str_jitter}" for i in range(100) if f"{i}-{canal_str_jitter}" in X.columns]

        if cols_jitter:
            datos_jitter = row[cols_jitter].to_numpy(dtype=float)
            datos_jitter = np.where(datos_jitter == -15, np.nan, datos_jitter)

            jitter_means.append(np.nanmean(datos_jitter))
            jitter_stds.append(np.nanstd(datos_jitter))
        else:
            jitter_means.append(np.nan)
            jitter_stds.append(np.nan)

        #  Calcular promedio y std del shimmer (solo canal activo)
        canal_str_shimmer = f"shimmer_standardized_{canal}"
        cols_shimmer = [f"{i}-{canal_str_shimmer}" for i in range(100) if f"{i}-{canal_str_shimmer}" in X.columns]

        if cols_shimmer:
            datos_shimmer = row[cols_shimmer].to_numpy(dtype=float)
            datos_shimmer = np.where(datos_shimmer == -15, np.nan, datos_shimmer)

            shimmer_means.append(np.nanmean(datos_shimmer))
            shimmer_stds.append(np.nanstd(datos_shimmer))
        else:
            shimmer_means.append(np.nan)
            shimmer_stds.append(np.nan)


        # Calcular promedio y std del log (solo canal activo)
        canal_str_log = f"logHNR_standardized_{canal}"
        cols_log = [f"{i}-{canal_str_log}" for i in range(100) if f"{i}-{canal_str_log}" in X.columns]

        if cols_log:
            datos_log = row[cols_log].to_numpy(dtype=float)
            datos_log = np.where(datos_log == -15, np.nan, datos_log)

            log_means.append(np.nanmean(datos_log))
            log_stds.append(np.nanstd(datos_log))
        else:
            log_means.append(np.nan)
            log_stds.append(np.nan)


         # Calcular promedio y std del intensity (solo canal activo)
        canal_str_intensity = f"intensity_standardized_{canal}"
        cols_intensity = [f"{i}-{canal_str_intensity}" for i in range(100) if f"{i}-{canal_str_intensity}" in X.columns]    
        if cols_intensity:
            datos_intensity = row[cols_intensity].to_numpy(dtype=float)
            datos_intensity = np.where(datos_intensity == -15, np.nan, datos_intensity)

            intensity_means.append(np.nanmean(datos_intensity))
            intensity_stds.append(np.nanstd(datos_intensity))
        else:
            intensity_means.append(np.nan)
            intensity_stds.append(np.nan)
        


        # Calcular slopes de pitch
        for cantidad_de_ms in cantidades_de_ms:
            if canal not in ["A", "B"]:
                slopes_dict[cantidad_de_ms].append(np.nan)
                regresiones_dict[cantidad_de_ms].append(None)
                continue

            start, end = 100 - cantidad_de_ms, 100
            cols_range = [f"{i}-{canal_str}" for i in range(start, end) if f"{i}-{canal_str}" in X.columns]

            if not cols_range:
                slopes_dict[cantidad_de_ms].append(np.nan)
                regresiones_dict[cantidad_de_ms].append(None)
                continue

            datos_y = row[cols_range].to_numpy(dtype=float)
            datos_y = np.where(datos_y == -15, np.nan, datos_y)
            mask = ~np.isnan(datos_y)
            datos_y = datos_y[mask]
            datos_x = np.arange(len(cols_range))
            datos_x = datos_x[mask].reshape(-1, 1)

            #print(datos_x)
            

            if len(datos_y) < 2:
                #print(datos_y)
                #print(datos_x)
                slopes_dict[cantidad_de_ms].append(np.nan)
                regresiones_dict[cantidad_de_ms].append(None)
                continue

            modelo_lineal = LinearRegression()
            ransac = RANSACRegressor(estimator=modelo_lineal, random_state=0)

            try:
                ransac.fit(datos_x, datos_y)
                slope = ransac.estimator_.coef_[0]
            except Exception:
                slope = np.nan
                ransac = None

            slopes_dict[cantidad_de_ms].append(slope)
            regresiones_dict[cantidad_de_ms].append(ransac)


        # Calcular slopes de intensity
        for cantidad_de_ms in cantidades_de_ms:
            if canal not in ["A", "B"]:
                slopes_dict_intensity[cantidad_de_ms].append(np.nan)
                regresiones_dict_intensity[cantidad_de_ms].append(None)
                continue

            start, end = 100 - cantidad_de_ms, 100
            cols_range = [f"{i}-{canal_str_intensity}" for i in range(start, end) if f"{i}-{canal_str_intensity}" in X.columns]

            if not cols_range:
                slopes_dict_intensity[cantidad_de_ms].append(np.nan)
                regresiones_dict_intensity[cantidad_de_ms].append(None)
                continue

            datos_y = row[cols_range].to_numpy(dtype=float)
            datos_y = np.where(datos_y == -15, np.nan, datos_y)
            mask = ~np.isnan(datos_y)
            datos_x = np.arange(len(cols_range))
            datos_x = datos_x[mask].reshape(-1, 1)


            if len(datos_y) < 2:
                slopes_dict_intensity[cantidad_de_ms].append(np.nan)
                regresiones_dict_intensity[cantidad_de_ms].append(None)
                continue

            modelo_lineal = LinearRegression()
            ransac = RANSACRegressor(estimator=modelo_lineal, random_state=0)

            try:
                ransac.fit(datos_x, datos_y)
                slope = ransac.estimator_.coef_[0]
            except Exception:
                slope = np.nan
                ransac = None

            slopes_dict_intensity[cantidad_de_ms].append(slope)
            regresiones_dict_intensity[cantidad_de_ms].append(ransac) 


    # Agregar columnas al DataFrame
 
    # Nuevas columnas de promedio y desviación estándar
    X["pitch_mean"] = pitch_means
    X["pitch_std"] = pitch_stds
    X["jitter_mean"] = jitter_means
    X["jitter_std"] = jitter_stds
    X["shimmer_mean"] = shimmer_means
    X["shimmer_std"] = shimmer_stds
    X["logHNR_mean"] = log_means
    X["logHNR_std"] = log_stds
    X["intensity_mean"] = intensity_means
    X["intensity_std"] = intensity_stds

    print("Columnas de jitter, shimmer y logHNR agregadas.")
    print("Columnas 'pitch_mean' y 'pitch_std' agregadas.")

    for cantidad_de_ms in cantidades_de_ms:
        col_name = f"slope_pitch_{cantidad_de_ms}"
        X[col_name] = slopes_dict[cantidad_de_ms]
        print(f"Columna '{col_name}' agregada.")

        col_name_intensity = f"slope_intensity_{cantidad_de_ms}"
        X[col_name_intensity] = slopes_dict_intensity[cantidad_de_ms]
        print(f"Columna '{col_name_intensity}' agregada.")

    conteo = Counter(y)
    print(conteo)

    # ======================
    # Guardar CSV y modelos
    # ======================
    if tipo == "overlap":
        X.to_csv("X_Y/overlap/dev/X.csv", index=True)
    else:
        X.to_csv("X_Y/no_overlap/dev/X.csv", index=True)
