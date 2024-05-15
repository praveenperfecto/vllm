import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

class T5ForVLLMWithCache(torch.nn.Module):
    def __init__(self, model_dir, tokenizer_dir=None, cache_size=5000):
        super(T5ForVLLMWithCache, self).__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_dir)
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_dir) if tokenizer_dir else None
        self.cache = {}

    def forward(self, text_inputs, max_length=512, return_tensors='pt'):
        # Tokenization and input preparation
        inputs = self.tokenizer(text_inputs, return_tensors=return_tensors, max_length=max_length, truncation=True, padding="max_length")
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        # Check cache
        key = tuple(input_ids.tolist())
        if key in self.cache:
            return self.cache[key]

        # Model execution
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        self.cache[key] = outputs.last_hidden_state
        return outputs.last_hidden_state

    def generate(self, input_text, max_length=512):
        # Tokenize the input and prepare for model input
        encoded_input = self.tokenizer(input_text, return_tensors="pt", max_length=max_length, truncation=True, padding="max_length")
        input_ids = encoded_input['input_ids']
    
        # Convert input_ids tensor to a tuple to be used as a dictionary key
        key = tuple(input_ids.view(-1).tolist())  # Flatten and convert to tuple
    
        # Check cache
        if key in self.cache:
            output_sequences = self.cache[key]
        else:
            # Generate sequences if not in cache
            output_sequences = self.model.generate(input_ids=input_ids, attention_mask=encoded_input['attention_mask'], max_length=max_length)
            self.cache[key] = output_sequences  # Store generated sequences in cache
    
        # Decode each generated sequence into text
        return [self.tokenizer.decode(g, skip_special_tokens=True) for g in output_sequences]
    
