from transformers import pipeline

def classify_topic(text, candidate_labels):
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    result = classifier(text, candidate_labels)
    return result['labels'][0]

# Contoh penggunaan:
if __name__ == "__main__":
    text = "Saya ingin membuka bisnis franchise minuman kekinian"
    labels = ["pendidikan", "kesehatan", "teknologi", "bisnis"]
    topik = classify_topic(text, labels)
    print(f"Topik: {topik}")
