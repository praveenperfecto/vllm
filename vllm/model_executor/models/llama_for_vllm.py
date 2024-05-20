import torch
from .cus_llama_7b.model import Transformer, ModelArgs
from .cus_llama_7b.inference import LLaMA
from sentencepiece import SentencePieceProcessor

class LLaMAForVLLM(torch.nn.Module):
    def __init__(self, checkpoints_dir, tokenizer_path, device='cpu'):
        super(LLaMAForVLLM, self).__init__()
        self.device = device
        self.tokenizer = SentencePieceProcessor()
        self.tokenizer.load(tokenizer_path)
        model_args = ModelArgs(device=self.device)  # Update parameters as necessary
        self.model = LLaMA.build(checkpoints_dir, tokenizer_path, True, 1024, 1, self.device)

    def forward(self, text_inputs):
        encoded_inputs = [self.tokenizer.encode(input_text, out_type=int) for input_text in text_inputs]
        input_tensor = torch.tensor(encoded_inputs, dtype=torch.long, device=self.device)
        output = self.model.model(input_tensor)
        return output

    def generate(self, input_text, max_length=512):
        token_ids = self.tokenizer.encode(input_text, out_type=int)
        input_tensor = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        generated_ids = self.model.text_completion([input_text], max_gen_len=max_length)[1][0]
        # Remove special tokens manually if necessary
        # For example, if pad_token_id = 0
        pad_token_id = self.tokenizer.pad_id()
        filtered_ids = [id for id in generated_ids if id != pad_token_id]
        return self.tokenizer.decode(filtered_ids)