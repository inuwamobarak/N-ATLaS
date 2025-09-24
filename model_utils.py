import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
from config import MODEL_NAME, GENERATION_CONFIG
import os
from dotenv import load_dotenv

load_dotenv()

class NAtlasModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.float16,
            device_map="auto"
        )

    def _format_messages(self, messages):
        current_date = datetime.now().strftime('%d %b %Y')
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            date_string=current_date,
        )

    def chat(self, messages):
        text = self._format_messages(messages)
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            **GENERATION_CONFIG
        )

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]