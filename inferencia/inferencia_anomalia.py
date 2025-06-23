import pandas as pd
from .utils_anomalia import carregar_modelos, inferencia_anomalia

def rodar_pipeline_anomalia(df_comportamento):
    modelos = carregar_modelos()
    df_final = inferencia_anomalia(df_comportamento, modelos)
    return df_final
