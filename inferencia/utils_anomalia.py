import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import euclidean_distances
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='xgboost')

def carregar_modelos():
    modelos = {
        "scaler": joblib.load("modelos/scaler.pkl"),
        "colunas_scaler": joblib.load("modelos/colunas_scaler.pkl"),
        "encoder_model": load_model("modelos/modelo_encoder.keras", compile=False),
        "autoencoder": load_model("modelos/modelo_autoencoder.keras", compile=False),
        "kmeans": joblib.load("modelos/kmeans_auto.pkl"),
        "encoder_tipo": joblib.load("modelos/encoder_tipo_transacao.pkl"),
        "encoder_semana": joblib.load("modelos/encoder_semana.pkl"),
        "encoder_horario": joblib.load("modelos/encoder_horario.pkl"),
        "modelo_xgb": joblib.load("modelos/modelo_xgb.pkl")
    }
    return modelos

def calcular_erros_e_distancias(df, modelos):
    colunas_scaler = modelos["colunas_scaler"]
    X_escalado = modelos["scaler"].transform(df[colunas_scaler])
    reconstruido = modelos["autoencoder"].predict(X_escalado)
    df["erro_reconstrucao"] = np.mean(np.square(X_escalado - reconstruido), axis=1)

    cod_latente = modelos["encoder_model"].predict(X_escalado)
    dist = euclidean_distances(cod_latente, modelos["kmeans"].cluster_centers_)
    df["distancia_cluster"] = np.min(dist, axis=1)
    return df

def gerar_motivo_alerta(row):
    motivos = []
    if row['modelo_predito'] == 1:
        motivos.append("modelo")
    if row['erro_reconstrucao'] > 0.1:
        motivos.append("erro alto")
    if row['distancia_cluster'] > 10:
        motivos.append("distância alta")
    if row['regra_valor_alto'] == 1:
        motivos.append("valor alto")
    if row['regra_horario'] == 1:
        motivos.append("horário suspeito")
    if row['regra_frequencia'] == 1:
        motivos.append("frequência alta")
    if row['regra_cluster'] == 1:
        motivos.append("desvio do cluster")
    return ", ".join(motivos) if motivos else "sem alerta"

def gerar_score_continuo(df):
    df['score_final'] = (
        df['modelo_predito'] * 0.5 +
        df['regra_valor_alto'] * 0.2 +
        df['regra_horario'] * 0.2 +
        df['regra_frequencia'] * 0.1
    )
    df['score_final'] = df['score_final'] / df['score_final'].max()

    df['faixa_risco'] = pd.cut(
        df['score_final'],
        bins=[-0.01, 0.4, 0.7, 1.0],
        labels=['baixo', 'moderado', 'alto']
    )
    return df

def inferencia_anomalia(df, modelos):
    df['transacao_data'] = pd.to_datetime(df['transacao_data'], errors='coerce')
    df = df.sort_values(['conta_id', 'transacao_data'])
    df = calcular_erros_e_distancias(df, modelos)

    df['tempo_desde_ultima'] = df.groupby('conta_id')['transacao_data'].diff().dt.total_seconds()
    df['regra_valor_alto'] = (df['transacao_valor'] > (df['media_valor'] + 3 * df['std_valor'])).astype(int)
    df['regra_horario'] = (df['faixa_horaria_Madrugada'] == 1).astype(int)
    df['regra_frequencia'] = (df['tempo_desde_ultima'] < 60).fillna(False).astype(int)
    df['regra_cluster'] = df.get('suspeita_cluster', '').isin(['baixa', 'media', 'alta']).astype(int)

    df['pontuacao_fraude'] = (
        2 * df['regra_cluster'] +
        2 * df['regra_horario'] +
        1 * df['regra_valor_alto'] +
        1 * df['regra_frequencia']
    )
    df['anomalia_confirmada'] = (df['pontuacao_fraude'] >= 3).astype(int)

    n_positivos = df['anomalia_confirmada'].sum()
    n_negativos = len(df) - n_positivos
    n_ruido = int(0.005 * len(df))

    positivos_idx = df[df['anomalia_confirmada'] == 1].sample(n=min(n_ruido, n_positivos), random_state=42).index
    df.loc[positivos_idx, 'anomalia_confirmada'] = 0

    negativos_idx = df[df['anomalia_confirmada'] == 0].sample(n=min(n_ruido, n_negativos), random_state=42).index
    df.loc[negativos_idx, 'anomalia_confirmada'] = 1

    colunas_validas = [
        'transacao_valor', 'fim_de_semana',
        'transacao_tipo_pix', 'transacao_tipo_transferencia',
        'erro_reconstrucao', 'distancia_cluster', 'mesma_titularidade',
        'faixa_horaria_Madrugada', 'dia_de_semana_Sabado', 'dia_de_semana_Domingo'
    ]
    df['modelo_predito'] = (modelos['modelo_xgb'].predict_proba(df[colunas_validas])[:, 1] >= 0.6).astype(int)

    df['regra_alerta'] = (
        (df['transacao_valor'] > 0.8) &
        (df['fim_de_semana'] == 1) &
        (df['mesma_titularidade'] == 0) &
        (df.get('faixa_horaria_Madrugada', 0) == 1)
    )
    df['decisao_final'] = ((df['modelo_predito'] == 1) | (df['regra_alerta'])).astype(int)

    df['nivel_suspeita'] = np.select(
        [
            (df['erro_reconstrucao'] > 0.2) & (df['distancia_cluster'] > 15),
            (df['erro_reconstrucao'] > 0.1) | (df['distancia_cluster'] > 10),
            (df['modelo_predito'] == 1)
        ],
        ['alta', 'media', 'baixa'],
        default='nenhuma'
    )

    df['risco_critico'] = (
        (df['anomalia_confirmada'] == 1) &
        (df['decisao_final'] == 0) &
        ((df['erro_reconstrucao'] > 0.1) | (df['distancia_cluster'] > 10))
    ).astype(int)

    df['motivo_alerta'] = df.apply(gerar_motivo_alerta, axis=1)
    df = gerar_score_continuo(df)

    return df