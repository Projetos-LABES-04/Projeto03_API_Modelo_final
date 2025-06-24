import pandas as pd
from .utils_anomalia import carregar_modelos, inferencia_anomalia

# Este módulo inclui logs controlados via log(), para facilitar depuração sem poluir o terminal
def log(msg):
    print(f"[LOG] {msg}")

def rodar_pipeline_anomalia(df_comportamento):
    log(f"Iniciando pipeline de anomalia com shape: {df_comportamento.shape}")
    
    modelos = carregar_modelos()

    df_final = inferencia_anomalia(df_comportamento, modelos)
    log(f"Inferência de anomalias concluída, shape final: {df_final.shape}")

    return df_final
