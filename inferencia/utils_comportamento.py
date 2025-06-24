import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAMINHO_MODELOS = os.path.join(BASE_DIR, "..", "modelos")

def carregar_modelos():
    print("Carregando modelos...")
    modelos = {
        "scaler": joblib.load(os.path.join(CAMINHO_MODELOS, "scaler.pkl")),
        "colunas_scaler": joblib.load(os.path.join(CAMINHO_MODELOS, "colunas_scaler.pkl")),
        "encoder_model": load_model(os.path.join(CAMINHO_MODELOS, "modelo_encoder.keras"), compile=False),
        "autoencoder": load_model(os.path.join(CAMINHO_MODELOS, "modelo_autoencoder.keras"), compile=False),
        "kmeans": joblib.load(os.path.join(CAMINHO_MODELOS, "kmeans_auto.pkl")),
        "encoder_tipo": joblib.load(os.path.join(CAMINHO_MODELOS, "encoder_tipo_transacao.pkl")),
        "encoder_semana": joblib.load(os.path.join(CAMINHO_MODELOS, "encoder_semana.pkl")),
        "encoder_horario": joblib.load(os.path.join(CAMINHO_MODELOS, "encoder_horario.pkl")),
        "modelo_xgb": joblib.load(os.path.join(CAMINHO_MODELOS, "modelo_xgb.pkl"))
    }
    print("Modelos carregados com sucesso!")
    return modelos

def preprocessar_transacoes(df: pd.DataFrame, modelos: dict) -> pd.DataFrame:
    print("Iniciando pré-processamento das transações...")
    df['transacao_data'] = pd.to_datetime(df['transacao_data'])

    dias_semana = {0:'Segunda', 1:'Terca', 2:'Quarta', 3:'Quinta', 4:'Sexta', 5:'Sabado', 6:'Domingo'}
    df['dia_de_semana'] = df['transacao_data'].dt.weekday.map(dias_semana)
    df['fim_de_semana'] = df['dia_de_semana'].isin(['Sabado', 'Domingo']).astype(int)

    def categorizar_hora(h):
        h = h.hour
        return 'Madrugada' if h < 6 else 'Manhã' if h < 12 else 'Tarde' if h < 18 else 'Noite'
    df['faixa_horaria'] = df['transacao_data'].dt.time.apply(categorizar_hora)

    df['mesma_titularidade'] = df['mesma_titularidade'].astype(int)

    def aplicar_encoder(df, coluna, encoder):
        arr = encoder.transform(df[[coluna]])
        nomes = encoder.get_feature_names_out([coluna])
        return pd.DataFrame(arr, columns=nomes, index=df.index)

    df_tipo = aplicar_encoder(df, 'transacao_tipo', modelos['encoder_tipo'])
    df_semana = aplicar_encoder(df, 'dia_de_semana', modelos['encoder_semana'])
    df_hora = aplicar_encoder(df, 'faixa_horaria', modelos['encoder_horario'])

    df_proc = pd.concat([
        df.drop(columns=['transacao_tipo', 'dia_de_semana', 'faixa_horaria'], errors='ignore'),
        df_tipo, df_semana, df_hora
    ], axis=1)

    dados_filtrados = df_proc[modelos['colunas_scaler']]
    df_proc[modelos['colunas_scaler']] = modelos['scaler'].transform(dados_filtrados)

    print("Pré-processamento concluído.")
    return df_proc

def detectar_anomalias(df_proc, modelos):
    print("Detectando padrões de comportamento e anomalias...")
    reconstruido = modelos['autoencoder'].predict(df_proc[modelos['colunas_scaler']])
    erro_reconstrucao = np.mean(np.square(df_proc[modelos['colunas_scaler']] - reconstruido), axis=1)

    dados_latentes = modelos['encoder_model'].predict(df_proc[modelos['colunas_scaler']])
    labels = modelos['kmeans'].predict(dados_latentes)

    q75, q90, q95 = np.quantile(erro_reconstrucao, [0.75, 0.90, 0.95])
    condicoes = [erro_reconstrucao > q95, erro_reconstrucao > q90, erro_reconstrucao > q75]
    valores = ['alta', 'media', 'baixa']
    suspeita = np.select(condicoes, valores, default='nenhuma')

    df_proc['erro_reconstrucao'] = erro_reconstrucao
    df_proc['cluster_autoencoder'] = labels
    df_proc['suspeita'] = suspeita

    thresholds_cluster = df_proc.groupby("cluster_autoencoder")['erro_reconstrucao'].quantile(0.95).to_dict()

    def classificar_suspeita_cluster(row):
        threshold = thresholds_cluster[row['cluster_autoencoder']]
        erro = row['erro_reconstrucao']
        if erro > threshold * 1.5:
            return "alta"
        elif erro > threshold:
            return "media"
        elif erro > threshold * 0.5:
            return "baixa"
        else:
            return "nenhuma"

    df_proc['suspeita_cluster'] = df_proc.apply(classificar_suspeita_cluster, axis=1)
    print("Análise comportamental concluída.")
    return df_proc

def gerar_perfil_cliente(df_conta):
    perfil = {
        'media_valor': df_conta['transacao_valor'].mean(),
        'std_valor': df_conta['transacao_valor'].std(),
        'percentual_pix': df_conta.filter(like='transacao_tipo_pix').sum(axis=1).mean(),
        'percentual_transferencia': df_conta.filter(like='transacao_tipo_transferencia').sum(axis=1).mean(),
        'percentual_pagamento': df_conta.filter(like='transacao_tipo_pagamento').sum(axis=1).mean(),
        'percentual_saque': df_conta.filter(like='transacao_tipo_saque').sum(axis=1).mean(),
        'percentual_deposito': df_conta.filter(like='transacao_tipo_deposito').sum(axis=1).mean(),
        'percentual_fim_de_semana': df_conta['fim_de_semana'].mean(),
        'percentual_mesma_titularidade': df_conta['mesma_titularidade'].mean()
    }
    horario_cols = [c for c in df_conta.columns if c.startswith('faixa_horaria_')]
    if horario_cols:
        perfil['horario_mais_comum'] = df_conta[horario_cols].sum().idxmax().replace('faixa_horaria_', '')
    dia_cols = [c for c in df_conta.columns if c.startswith('dia_de_semana_')]
    if dia_cols:
        perfil['dia_semana_mais_comum'] = df_conta[dia_cols].sum().idxmax().replace('dia_de_semana_', '')
    return pd.Series(perfil)

def gerar_perfis(df_final):
    print("Gerando perfis médios por conta...")
    resultado = df_final.groupby('conta_id', group_keys=False).apply(gerar_perfil_cliente).reset_index()
    print("Perfis gerados com sucesso.")
    return resultado 
