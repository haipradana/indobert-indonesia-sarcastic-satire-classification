import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model'))

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)  

model.eval()

def model_predict(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    return 'sarcasm' if prediction == 1 else 'not sarcasm'

def predict():
    text = [
        "Dia saking pinternya gatau cara bagi waktu",
        "Kamu penulis ya? pandai sekali mengarang cerita",
        "kamu cerdas banget",
        "dia emang cakep orangnya"
    ]
    for i, text in enumerate(text, 1):
        predicted_label = model_predict(text)
        print(f"{i}. Text: '{text}' -> Predicted: {predicted_label}")
        
if __name__ == "__main__":
    predict()
    
'''
Output:
1. Text: 'Dia saking pinternya gatau cara bagi waktu' -> Predicted: sarcasm
2. Text: 'Kamu penulis ya? pandai sekali mengarang cerita' -> Predicted: sarcasm
3. Text: 'kamu cerdas banget' -> Predicted: not sarcasm
4. Text: 'dia emang cakep orangnya' -> Predicted: not sarcasm
'''