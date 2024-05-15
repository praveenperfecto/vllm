import torch
from transformers import T5Model, T5Tokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer


class T5ForVLLM(torch.nn.Module):

    def __init__(self, model_dir, tokenizer_dir=None):
        super(T5ForVLLM, self).__init__()
        # Load the model
        self.model = T5ForConditionalGeneration.from_pretrained(model_dir)
        self.tokenizer = None

        # Load tokenizer if a directory is provided
        if tokenizer_dir:
            self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_dir)

    def forward(self, text_inputs, max_length=512, return_tensors='pt'):
        if self.tokenizer:
            # Encode the inputs
            inputs = self.tokenizer(text_inputs, return_tensors=return_tensors, max_length=max_length, truncation=True, padding="max_length")
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask
        else:
            # Assume input_ids and attention_mask are provided directly if tokenizer is not used
            input_ids = text_inputs.get('input_ids')
            attention_mask = text_inputs.get('attention_mask')

        # Generate model outputs
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state


    def generate(self, input_text, max_length=512, **kwargs):
        # Check if the tokenizer is loaded
        if not self.tokenizer:
            raise ValueError("Tokenizer is not loaded. Cannot generate text without tokenizer.")

        # Encode the input text
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=max_length, truncation=True, padding="max_length")
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # Generate output from the model, passing through any additional kwargs
        output_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length, **kwargs)
        
        # Decode generated output into text
        return [self.tokenizer.decode(generated_id, skip_special_tokens=True) for generated_id in output_ids]
