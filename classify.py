from processor_regex import classify_with_regex
from processor_bert import classify_with_bert
from processor_llm import classify_with_llm
import pandas as pd


def classify(logs):
    labels = []
    for source, log_msg in logs:
        label = classify_log(source, log_msg)
        labels.append(label)
    return labels


def classify_log(source,log_message):
    if source == "LegacyCRM":
        label = classify_with_llm(log_message)
    else:
        label = classify_with_regex(log_message)
        if label is None:
            label = classify_with_bert(log_message)
        return label
    

def classify_csv(input_file):
    df = pd.read_csv(input_file)
    df["target_label"] = classify(list(zip(df["source"], df["log_message"])))

    output_file = "resources/output.csv"
    df.to_csv(output_file, index=False)



if __name__ == "__main__":
    classify_csv("resources/test.csv")


    # logs= [
    #     ("ModernCRM", "IP 192.168.1.123 blocked due to potential attack."),
    #     ("BillingSystem", "User User123 logged in."),
    #     ("AnalyticsEngine", "File file.txt uploaded successfully by user user123."),
    #     ("AnalyticsEngine", "backup started at 2023-10-01 12:00:00."),
    #     ("ModernHR", "GET /users/123 HTTP/1.1 200 OK"),
    #     ("ModernHR", "Admin acess escalation attempt detected for user User123."),
    #     ("LegacyCRM", "Disk cleanup completed successfully."),
    #     ("LegacyCRM", "System reboot initiated by user user123."),
    #     ("LegacyCRM", "Account with ID 123 created by user user123."),
    #     ("LegacyCRM", "Invoice #123456 generated successfully."),
    # ]

    # classified_logs = classify(logs)
    # print(classified_logs)