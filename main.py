from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from typing import List
from inferencia.inferencia_comportamento import rodar_pipeline_comportamento
from inferencia.inferencia_anomalia import rodar_pipeline_anomalia
app = FastAPI(title="API Comportamento e Anomalia")

# Este módulo inclui logs controlados via log(), para facilitar depuração sem poluir o terminal
def log(msg):
    print(f"[LOG] {msg}")

# Modelo de entrada com os dados diretamente na requisição
class Transacao(BaseModel):
    transacao_id: str
    cliente_id: int
    conta_id: str
    conta_destino_id: str
    mesma_titularidade: bool
    transacao_data: str 
    transacao_valor: float
    transacao_tipo: str

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/inferencia")
def inferencia_completa(transacoes: List[Transacao]):
    try:
        # Converte lista de objetos para DataFrame
        log("Recebendo transações")
        df = pd.DataFrame([t.dict() for t in transacoes])
        log(f"DataFrame criado com shape: {df.shape}")

        # 1. Executa pipeline de comportamento
        log("Executando pipeline de comportamento")
        df_comportamento = rodar_pipeline_comportamento(df)
    
        # 2. Executa pipeline de anomalia
        log("Executando pipeline de anomalia")
        df_final = rodar_pipeline_anomalia(df_comportamento)

        # 3. Prepara resposta
        log("Pipeline de inferência concluído")
        total = len(df_final)
        anomalias = df_final['decisao_final'].sum()

        amostra = df_final[[
            'transacao_id', 'conta_id', 'decisao_final',
            'anomalia_confirmada', 'nivel_suspeita', 'motivo_alerta'
        ]].head(5).to_dict(orient='records')


        return {
            "total_transacoes": total,
            "anomalias_detectadas": int(anomalias),
            "amostra": amostra
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))