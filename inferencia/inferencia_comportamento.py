import pandas as pd
from .utils_comportamento import (
    carregar_modelos, preprocessar_transacoes,
    detectar_anomalias, gerar_perfis
)

# Este módulo inclui logs controlados via log(), para facilitar depuração sem poluir o terminal
def log(msg):
    print(f"[LOG] {msg}")

def rodar_pipeline_comportamento(df):
    log("Rodando pipeline_comportamento com DataFrame recebido")
    
    modelos = carregar_modelos()

    df_proc = preprocessar_transacoes(df, modelos)
    log(f"Dados preprocessados: {df_proc.shape}")

    df_final = detectar_anomalias(df_proc, modelos)
    log(f"Anomalias detectadas: {df_final.shape}")

    perfis = gerar_perfis(df_final)
    log(f"Perfis gerados: {perfis.shape}")

    df_completo = pd.merge(df_final, perfis, on="conta_id", how="left")
    log(f"Pipeline comportamento finalizado com shape: {df_completo.shape}")

    return df_completo
