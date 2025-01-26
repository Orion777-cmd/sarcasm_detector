import torch
from transformers import BertModel, BertTokenizer

# Define the custom model
class SarcasmDetectionModel(torch.nn.Module):
    def __init__(self, pretrained_model_name="bert-base-uncased", num_labels=2):
        super(SarcasmDetectionModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)  # Load BERT backbone
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_labels)  # Add classification head

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # Use the pooled output from BERT
        logits = self.classifier(pooled_output)  # Pass through the classification head
        return logits

# Load the trained model
model_path = "model/sarcasm_model.pt"  # Replace with the path to your trained model
model = SarcasmDetectionModel(pretrained_model_name="bert-base-uncased", num_labels=2)

# Load the state dictionary into the custom model
state_dict = torch.load(model_path, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()  # Set the model to evaluation mode

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def predict_sarcasm(text):
    """
    Predict sarcasm for the given text using the trained model.

    Args:
    - text: Input text string.

    Returns:
    - prediction: Predicted label (0 or 1).
    """
    # Tokenize and encode the input text
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    
    # Perform inference
    with torch.no_grad():
        logits = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
        prediction = torch.argmax(logits, dim=1).item()
    
    return prediction

# Example usage
example_text = "Oh, great. Another rainy day. How exciting."
print(f"Predicted label: {predict_sarcasm(example_text)}")
