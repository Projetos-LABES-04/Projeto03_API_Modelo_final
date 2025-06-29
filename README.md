# Projeto: Mapeamento de Comportamento com Autoencoder e KMeans

## Etapas Desenvolvidas

---

### 1. Pré-processamento e Enriquecimento dos Dados

A base original foi enriquecida com variáveis temporais derivadas da data da transação:

- Dia da semana (segunda a domingo)
- Faixa horária (madrugada, manhã, tarde, noite)
- Flag fim de semana

As variáveis categóricas foram transformadas em variáveis numéricas via OneHotEncoder, enquanto as variáveis numéricas foram normalizadas com RobustScaler.

---

### 2. Representação Comportamental com Autoencoder

O autoencoder é uma rede neural não supervisionada usada para aprender os padrões normais de transações.

- **Encoder**: comprime os dados em um vetor latente de 3 dimensões.
- **Decoder**: reconstrói a transação original.

O erro de reconstrução representa o quanto uma transação se desvia do padrão aprendido:

- Erros baixos → transações comuns
- Erros altos → transações anômalas

---

### 3. Clusterização com KMeans

Os vetores latentes foram agrupados com **KMeans (k=2)** para identificar grupos comportamentais distintos.  
Cada transação recebeu o rótulo `cluster_autoencoder`.

---

### Validação Estatística dos Clusters

Para confirmar que os clusters representam grupos distintos, foram aplicadas duas análises:

- **ANOVA**: indicou diferenças significativas (p < 0.0001) nas variáveis entre os clusters.  
- **GLM**: confirmou essas diferenças ajustando por fatores como horário e tipo de transação.

Ambas reforçam que os clusters refletem **comportamentos reais e distintos**.

---

### 4. Classificação de Suspeita

Flag baseada no erro de reconstrução:

- Alta suspeita: > p95
- Média suspeita: > p90
- Baixa suspeita: > p75
- Nenhuma: abaixo disso

---

### 5. Geração de Perfis Comportamentais

Para cada conta:

- Média de valor transacionado
- Frequência por tipo, dia da semana e horário
- Faixa horária e dia mais comuns

---

## Etapas Supervisionadas – Detecção de Anomalias

---

### 1. Construção do Rótulo `anomalia_confirmada`

Criado com base em regras:

- Valor alto
- Horário suspeito
- Frequência elevada
- Desvio do cluster (erro ou distância)

Transações com pontuação ≥ 3 receberam `anomalia_confirmada = 1`.

> Ruído artificial de 0,5% foi adicionado para simular incertezas.

---

### 2. Representação com Autoencoder e KMeans

Erro de reconstrução e distância ao centróide dos clusters foram usados como variáveis explicativas.

---

### 3. Treinamento do Modelo (XGBoost)

- Modelo: XGBoost
- Balanceamento: SMOTE
- Alvo: `anomalia_confirmada`

**Métricas**:

- F1-score
- Recall por grupo sensível
- Equality of Opportunity

---

### 4. Avaliação com Múltiplos Thresholds

Testados thresholds de 0.40 a 0.80:

| Threshold | F1-score | Falsos Negativos | Falsos Positivos | Equality of Opportunity |
|-----------|----------|------------------|------------------|--------------------------|
| 0.40      | 0.9057   | 85.916           | 3.288            | 0.0006                   |
| 0.50      | 0.9045   | 87.191           | 2.999            | 0.0006                   |
| 0.55      | 0.9040   | 87.657           | 2.964            | 0.0006                   |
| **0.60 ✅** | **0.9035** | **88.051**       | **2.962**         | **0.0005**               |
| 0.70      | 0.9025   | 88.903           | 2.924            | 0.0005                   |
| 0.80      | 0.9019   | 89.451           | 2.922            | 0.0004                   |

---

## Interpretação

- Thresholds baixos → mais detecção, mais falsos positivos
- Thresholds altos → menos falsos positivos, mais falsos negativos
- **Escolha final: threshold = 0.60** (melhor equilíbrio entre sensibilidade e robustez)

---

## 🔗 API – Endpoints

### `GET /health`  
Verifica se a API está online.  
Resposta: `{ "status": "ok" }`

### `POST /inferencia`  
Executa o pipeline completo (comportamento + anomalia) e retorna os resultados.

---
## Estrutura do Código

- `main.py`:  
  Define os endpoints da API (`/health` e `/inferencia`) e orquestra o pipeline de inferência.

- `inferencia/`:  
  Diretório com os pipelines:
  - `inferencia_comportamento.py`: aplica pré-processamento, codificação e clusterização (Autoencoder + KMeans)
  - `inferencia_anomalia.py`: aplica regras e o modelo XGBoost para classificar anomalias

- `modelos/`: 
  - `colunas_scaler.pkl` — colunas utilizadas no scaler original  
  - `scaler.pkl` — instância do `RobustScaler` treinado  
  - `encoder_horario.pkl` — OneHotEncoder para faixas horárias  
  - `encoder_semana.pkl` — OneHotEncoder para dias da semana  
  - `encoder_tipo_transacao.pkl` — OneHotEncoder para tipo de transação
  - `kmeans_auto.pkl` — modelo KMeans treinado sobre os vetores do Autoencoder
  - `modelo_encoder.keras` — parte encoder da rede (reduz dimensionalidade)  
  - `modelo_autoencoder.keras` — autoencoder completo usado para reconstrução
  - `modelo_xgb.pkl` — modelo XGBoost final para inferência de anomalias
---

## requirements.txt

Lista de bibliotecas necessárias para execução do projeto:

- `fastapi==0.111.0`  
- `uvicorn==0.30.0`
- `joblib==1.5.0`  
- `scikit-learn==1.6.1`  
- `xgboost==3.0.2`
- `tensorflow==2.19.0`
- `pandas==2.2.3`  
- `numpy==2.1.3`
- `python-dateutil==2.9.0.post0`
