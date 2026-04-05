from transformers import AutoModelForCausalLM, AutoTokenizer
from config import Config
import torch

class AITutorModel:
    def __init__(self):
        print(f"Loading model {Config.MODEL_NAME} on {Config.DEVICE}...")
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(Config.MODEL_NAME).to(Config.DEVICE)
        print("Model loaded successfully.")

    def generate_response(self, user_input: str) -> str:
        messages = [
            {"role": "system", "content": "You are a friendly and helpful Chinese grammar AI tutor. You explain Chinese grammar simply and provide examples."},
            {"role": "user", "content": user_input}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(Config.DEVICE)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

# Initialize the model as a singleton so it is loaded once on server startup.
try:
    tutor_model = AITutorModel()
except Exception as e:
    print(f"Warning: Failed to load model on startup -> {str(e)}")
    tutor_model = None
