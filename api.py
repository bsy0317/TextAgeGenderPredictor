from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer
from models.gender_age_model import GenderAgeModel
from utils.checkpoint import load_checkpoint

app = Flask(__name__)

device_name = "cpu"
if torch.cuda.is_available():
    device_name = "cuda"
elif torch.backends.mps.is_available():
    device_name = "mps"
device = torch.device(device_name)
tokenizer = AutoTokenizer.from_pretrained('beomi/KcELECTRA-base')

model = GenderAgeModel().to(device)
checkpoint_path = "./model/checkpoint.pth"
optimizer = torch.optim.Adam(model.parameters()) 
start_epoch = load_checkpoint(model, optimizer, checkpoint_path)
model.eval()

def predict(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
    
    with torch.no_grad():
        age_logits, gender_logits = model(inputs['input_ids'], inputs['attention_mask'])
    
    age_prediction = torch.argmax(age_logits, dim=1).item()
    gender_prediction = torch.argmax(gender_logits, dim=1).item()
    
    age_map = {0: "20s", 1: "30s", 2: "40s", 3: "50s", 4: "60s", 5: "70s"}
    gender_map = {0: "Male", 1: "Female"}
    print(age_prediction, gender_prediction)
    
    return {"age": age_map[age_prediction], "gender": gender_map[gender_prediction]}

@app.route('/predict', methods=['POST'])
def predict_text():
    data = request.get_json()
    text = data.get('text')
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    prediction = predict(text)
    return jsonify(prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
