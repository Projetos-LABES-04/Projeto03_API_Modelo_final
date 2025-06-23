import pandas as pd
from .utils_comportamento import (
    carregar_modelos, preprocessar_transacoes,
    detectar_anomalias, gerar_perfis
)

def rodar_pipeline_comportamento(caminho_csv):
    modelos = carregar_modelos()
    df = pd.read_csv(caminho_csv)
    df_proc = preprocessar_transacoes(df, modelos)
    df_final = detectar_anomalias(df_proc, modelos)
    perfis = gerar_perfis(df_final)
    df_completo = pd.merge(df_final, perfis, on="conta_id", how="left")
    return df_completo
