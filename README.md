## How to use this model?


```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model
tokenizer = AutoTokenizer.from_pretrained("indobert-indonesia-sarcastic-satire-classification/model")
model = AutoModelForSequenceClassification.from_pretrained("indobert-indonesia-sarcastic-satire-classification/model")

# Predict
def predict(text: str):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    return 'sarcasm' if prediction == 1 else 'not sarcasm'

# Example
result = predict("Beda, ini inovasi terbaru, Apple kan king of innovation")
print(result) #output = sarcasm
```

### Or just using the script in the GitHub Repos

```bash
cd scripts
python predict.py
```

## Evaluation Results

The model was evaluated on the test set with the following metrics:

| Metric     | Value   |
|------------|---------|
| Accuracy   | 0.8378  |
| Precision  | 0.8405  |
| Recall     | 0.8286  |
| F1-Score   | 0.8345  |

### Training History

| Epoch | Train Loss | Val Loss | Accuracy | Precision | F1-Score | Recall  |
|-------|------------|----------|----------|-----------|----------|---------|
| 1     | 0.4559     | 0.3512   | 0.8409   | 0.9022    | 0.8261   | 0.7618  |
| 2     | 0.2491     | 0.3924   | 0.8339   | 0.7835    | 0.8459   | 0.9190  |
| 3     | 0.1198     | 0.5980   | 0.8429   | 0.8188    | 0.8471   | 0.8774  |
| 4     | 0.0439     | 0.9497   | 0.8444   | 0.8231    | 0.8479   | 0.8742  |
| 5     | 0.0097     | 0.9962   | 0.8522   | 0.8421    | 0.8529   | 0.8640  |
