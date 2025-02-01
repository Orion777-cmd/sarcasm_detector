import torch
from transformers import BertModel, BertTokenizer

class SarcasmDetectionModel(torch.nn.Module):
    def __init__(self, pretrained_model_name="bert-base-uncased", num_labels=2):
        super(SarcasmDetectionModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output 
        logits = self.classifier(pooled_output)  
        return logits

model_path = "model/sarcasm_model.pt"  
model = SarcasmDetectionModel(pretrained_model_name="bert-base-uncased", num_labels=2)


state_dict = torch.load(model_path, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def predict_sarcasm(text):
    """
    Predict sarcasm for the given text using the trained model.

    Args:
    - text: Input text string.

    Returns:
    - prediction: Predicted label (0 or 1).
    """

    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    
    with torch.no_grad():
        logits = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
        prediction = torch.argmax(logits, dim=1).item()
    
    return prediction

example_text = "Oh, great. Another rainy day. How exciting."
print(f"Predicted label: {predict_sarcasm(example_text)}")
