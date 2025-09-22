from typing import Dict
import torch
import numpy as np
from transformers import (
    GPT2Tokenizer,
    GPT2Model,
    BertTokenizer,
    BertModel,
    GemmaTokenizer,
    GemmaModel,
    AutoTokenizer,
    AutoModel,
)


class AttentionExtractor:

    def __init__(self, model_name: str = "gpt2"):
        print(f"Loading model: {model_name}")
        self.model_name = model_name

        if "gpt2" in model_name.lower():
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.model = GPT2Model.from_pretrained(model_name, output_attentions=True)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        elif "bert-large-uncased" in model_name.lower():
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertModel.from_pretrained(model_name, output_attentions=True)
        elif "google/gemma-2b-it" in model_name.lower():
            self.tokenizer = GemmaTokenizer.from_pretrained(model_name)
            self.model = GemmaModel.from_pretrained(model_name, output_attentions=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name, output_attentions=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()

        print("Model loaded successfully")

    def extract_attention(self, text: str) -> Dict:
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        with torch.no_grad():
            outputs = self.model(**inputs)

        attentions = outputs.attentions

        attention_data = []
        for layer_attention in attentions:
            # reshaping from [batch_size, num_heads, seq_len, seq_len] -> [num_heads, seq_len, seq_len]
            layer_attn = layer_attention[0].detach().numpy()
            attention_data.append(layer_attn)

        return {
            "tokens": tokens,
            "attention": attention_data,
            "num_layers": len(attention_data),
            "num_heads": attention_data[0].shape[0],
            "seq_len": len(tokens),
        }

    def get_attn_matrix(
        self, attention_data: Dict, layer: int, head: int
    ) -> np.ndarray:
        return attention_data["attention"][layer][head]
