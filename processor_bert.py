from sentence_transformers import SentenceTransformer

import joblib

transformer_model = SentenceTransformer('all-MiniLM-L6-v2')

classifier_model = joblib.load('models/log_classifier.joblib')

def classify_with_bert(log_message):
    embedding = transformer_model.encode(log_message)
    probabilities = classifier_model.predict_proba([embedding])[0]
    if max (probabilities) < 0.5:
        return "Unclassified"
    predicted_label = classifier_model.predict([embedding])[0]

    return predicted_label 

if __name__ == "__main__":
    logs = [
        "GET /users/123 HTTP/1.1 200 OK",
        "User User123 logged in.",
        "Backup started at 2023-10-01 12:00:00.",
        "Backup completed successfully.",
        "System updated to version 1.2.3.",
        "File file.txt uploaded successfully by user user123.",
        "Disk cleanup completed successfully.",
        "System reboot initiated by user user123.",
        "Account with ID 123 created by user user123.",
        "Opa pessoal, tudo bom?"
    ]
    for log in logs:
        label = classify_with_bert(log)
        print(log, "->", label)