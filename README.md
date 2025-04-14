# Sistema Inteligente de Classificação de Logs

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-BERT%2C%20Llama3-orange)
![License](https://img.shields.io/badge/License-MIT-green)

Solução completa para classificação automática de logs utilizando técnicas combinadas de RegEx, LLMs e modelos de transformadores.

## 🔍 Visão Geral
```python
# Exemplo de fluxo completo
from classify import classify_csv
classify_csv("resources/production_logs.csv")  # Gera output.csv classificado

🛠 Configuração

# requirements.txt
sentence-transformers==4.0.2
scikit-learn==1.6.1
groq==0.22.0
python-dotenv==1.1.0
pandas==2.2.3

Modelos Pré-treinados
# processor_bert.py
transformer_model = SentenceTransformer('all-MiniLM-L6-v2')  # Embeddings de 384 dimensões
classifier_model = joblib.load('models/log_classifier.joblib')  # Modelo Logistic Regression

🧩 Módulos Principais
1. Classificador RegEx

# processor_regex.py
PATTERNS = {
    r"User User\d+ logged (in|out).": "User Action",
    r"Backup (started|ended) at .*": "System Notification",
    # +12 padrões otimizados
}

def classify_with_regex(log_msg: str) -> str | None:
    for pattern, label in PATTERNS.items():
        if re.search(pattern, log_msg, re.IGNORECASE):
            return label
    return None

2. Classificador Llama-3 via Groq

# processor_llm.py
def classify_with_llm(log_msg: str) -> str:
    prompt = f'''Classifique em: (1) Workflow Error, (2) Deprecation Warning.
    Mensagem: {log_msg}'''
    
    response = groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

3. Pipeline BERT
# processor_bert.py
def classify_with_bert(log_msg: str) -> str:
    embedding = transformer_model.encode(log_msg)  # Vetorização
    proba = classifier_model.predict_proba([embedding])[0]
    
    if max(proba) < 0.5:  # Threshold de confiança
        return "Unclassified"
    return classifier_model.predict([embedding])[0]

📈 Pipeline de Treinamento
# training.ipynb
# 1. Geração de Embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['log_message'])

# 2. Clusterização Semântica
dbscan = DBSCAN(eps=0.2, min_samples=1, metric='cosine')
df['cluster'] = dbscan.fit_predict(embeddings)

# 3. Treino do Classificador
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(embeddings, df['target_label'])

# 4. Salvando Modelo
joblib.dump(logistic_model, 'models/log_classifier.joblib')

📊 Métricas de Desempenho
               Precision   Recall   F1-Score   Support
HTTP Status        1.00      1.00      1.00       304
Security Alert     1.00      0.99      1.00       123
Critical Error     0.91      1.00      0.95        48
System Notif.      0.98      0.97      0.98       217
User Action        0.99      0.99      0.99       189

Accuracy                               0.99       881
Macro Avg          0.98      0.99      0.98
Weighted Avg       0.99      0.99      0.99

🚨 Casos de Uso Complexos
# Logs não estruturados
log = "API response time exceeded 5s threshold"
print(classify_with_bert(log))  # "Performance Degradation"

# Logs híbridos
log = "Deprecated method 'get_legacy_data' called from module X"
print(classify_with_llm(log))  # "Deprecation Warning"

🌐 Arquitetura
    A[Input Logs] --> B{Source}
    B -->|LegacyCRM| C[LLM Classifier]
    B -->|Others| D[RegEx Check]
    D -->|Match| E[Return Label]
    D -->|No Match| F[BERT Classifier]
    F --> G{Confidence > 0.5?}
    G -->|Yes| H[Return Prediction]
    G -->|No| I[Return Unclassified]

🔄 Fluxo de Dados
# Exemplo completo de classificação
log_entry = ("AnalyticsEngine", "Database connection pool 80% utilized")

def classify_log(source, message):
    if source == "LegacyCRM":
        return classify_with_llm(message)
    regex_result = classify_with_regex(message)
    return regex_result if regex_result else classify_with_bert(message)

print(classify_log(*log_entry))  # "Resource Usage"