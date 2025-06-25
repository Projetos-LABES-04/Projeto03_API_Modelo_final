# Documentação

## **Projeto: Mapeamento de Comportamento com Autoencoder e KMeans**

---

## Etapas Desenvolvidas

### **1. Pré-processamento e Enriquecimento dos Dados**

A base original foi enriquecida com variáveis temporais derivadas da data da transação:

- **Dia da semana** (segunda a domingo)
- **Faixa horária** (madrugada, manhã, tarde, noite)
- **Flag fim de semana**

As variáveis categóricas foram transformadas com **OneHotEncoder**, e as numéricas normalizadas com **RobustScaler**.

---

### **2. Representação Comportamental com Autoencoder**

O **autoencoder** é uma rede neural não supervisionada para aprender padrões normais de transações.

- **Encoder**: reduz a transação para 3 dimensões latentes
- **Decoder**: tenta reconstruir a transação original

O **erro de reconstrução** indica o quanto a transação se desvia do padrão aprendido:

- Erros baixos → transações comuns  
- Erros altos → transações anômalas

---

### **3. Clusterização com KMeans**

Os vetores latentes foram agrupados com **KMeans (k=2)** para identificar **grupos de comportamento distintos**.  
Cada transação recebe um `cluster_autoencoder`.

---

### **4. Classificação de Suspeita**

Flag criada com base no erro de reconstrução:

- **Alta suspeita**: > p95  
- **Média suspeita**: > p90  
- **Baixa suspeita**: > p75  
- **Nenhuma**: caso contrário

---

### **5. Geração de Perfis Comportamentais**

Para cada conta foi gerado um **perfil médio**, com:

- Valor médio das transações
- Frequência por tipo (PIX, saque etc.)
- Frequência por horário e dia
- Faixa horária e dia da semana mais comuns

---

## Etapas Desenvolvidas – Detecção Supervisionada (Anomalias)

### **1. Construção do Rótulo de Anomalia (`fraude_confirmada`)**

Criado com base em regras:

- Valor muito alto  
- Horário de madrugada  
- Frequência alta  
- Desvio do cluster

Transações com pontuação ≥ 3 recebem `fraude_confirmada = 1`.

Foi adicionado **ruído de 0.5%** para simular incertezas.

---

### **2. Representação com Autoencoder e KMeans**

As variáveis `erro_de_reconstrucao` e `distancia_cluster` foram usadas no modelo supervisionado.

---

### **3. Treinamento do Modelo Supervisionado (XGBoost)**

- Modelo: **XGBoost**
- Balanceamento: **SMOTE**
- Alvo: `fraude_confirmada`
- Métricas: F1-score, recall por grupo e **Equality of Opportunity**

---

### **4. Avaliação com Múltiplos Thresholds**

Testados thresholds de **0.40 a 0.80**.  
O **threshold 0.60** foi escolhido como ideal.

---

### Resultados

| Threshold | F1-score | Falsos Negativos | Falsos Positivos | Equality of Opportunity |
|-----------|----------|------------------|------------------|--------------------------|
| 0.40      | 0.9057   | 85.916           | 3.288            | 0.0006                   |
| 0.50      | 0.9045   | 87.191           | 2.999            | 0.0006                   |
| 0.55      | 0.9040   | 87.657           | 2.964            | 0.0006                   |
| 0.60 ✅   | 0.9035   | 88.051           | 2.962            | 0.0005                   |
| 0.70      | 0.9025   | 88.903           | 2.924            | 0.0005                   |
| 0.80      | 0.9019   | 89.451           | 2.922            | 0.0004                   |

---

## Interpretação

- **Thresholds baixos** (ex: 0.40): mais detecção, mas mais falsos positivos
- **Thresholds altos** (ex: 0.80): menos falsos positivos, mas mais falsos negativos

### Escolha do Threshold Ideal

O projeto utilizou o **threshold 0.60** por equilibrar:

- Alta detecção (sensibilidade)
- Baixos falsos positivos (especificidade)
- Justiça entre grupos (`Equality of Opportunity ≈ 0.0005`)

---

## API de Inferência (FastAPI)

O pipeline foi encapsulado em uma **API REST com FastAPI**, permitindo inferência de comportamento + anomalias via HTTP.

---

### Endpoints Disponíveis

#### `GET /health`

Verifica se a API está online.

**Resposta:**
```json
{ "status": "ok" }
```

## API de Inferência (FastAPI)

### Endpoint: `POST /inferencia`

Recebe um lote de transações e retorna a inferência de comportamento e anomalia para cada uma.

---

### Exemplo de Requisição

```json
[
  {
    "transacao_id": "abc123",
    "cliente_id": 1,
    "conta_id": "conta001",
    "conta_destino_id": "conta002",
    "mesma_titularidade": true,
    "transacao_data": "2025-06-25T14:30:00",
    "transacao_valor": 150.75,
    "transacao_tipo": "pix"
  }
]
```

# API de Inferência – FastAPI

Esta API recebe um lote de transações e retorna o resultado da análise de comportamento + detecção de anomalias.

---

## Exemplo de Resposta

```json
{
  "total_transacoes": 100,
  "anomalias_detectadas": 5,
  "amostra": [
    {
      "transacao_id": "abc123",
      "conta_id": "conta001",
      "decisao_final": 1,
      "anomalia_confirmada": 1,
      "nivel_suspeita": "alta",
      "motivo_alerta": "erro alto, valor alto, cluster"
    }
  ]
}
```
## Como Executar a API

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

## requirements.txt

```txt
pandas
numpy
tensorflow
scikit-learn
joblib
statsmodels
imbalanced-learn
xgboost
matplotlib
fastapi
uvicorn
