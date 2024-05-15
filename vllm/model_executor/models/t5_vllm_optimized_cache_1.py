# File: t5_vllm_optimized_cache.py
import importlib
from typing import Dict, Optional, Type
import torch.nn as nn
import redis
import pickle
from .t5 import T5ForVLLM
from transformers import T5ForConditionalGeneration, T5Tokenizer
import numpy as np
import torch


# Initialize Redis client
redis_client = redis.Redis(host='localhost', port=6379, db=0)

class T5ForVLLMWithDistributedCache(nn.Module):
    def __init__(self, model_dir, tokenizer_dir=None, cache_size=5000):
        super(T5ForVLLMWithDistributedCache, self).__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_dir)
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_dir if tokenizer_dir else model_dir)
        self.cache_size = cache_size

    def _get_cache_key(self, input_ids):
        return str(input_ids.tolist())

    def _check_cache(self, key):
        cached_result = redis_client.get(key)
        return pickle.loads(cached_result) if cached_result else None

    def _add_to_cache(self, key, value):
        if redis_client.dbsize() > self.cache_size:
            redis_client.randomkey()
            redis_client.delete(key)  # Evict a random key
        redis_client.set(key, pickle.dumps(value))

    def forward(self, text_inputs, max_length=512, return_tensors='pt'):
        inputs = self.tokenizer(text_inputs, return_tensors=return_tensors, max_length=max_length, truncation=True, padding="max_length")
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        key = self._get_cache_key(input_ids)
        cached_result = self._check_cache(key)
        if cached_result:
            return cached_result

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        self._add_to_cache(key, outputs.last_hidden_state)
        return outputs.last_hidden_state

    # def generate(self, input_text, max_length=512):
    #     encoded_input = self.tokenizer(input_text, return_tensors="pt", max_length=max_length, truncation=True, padding="max_length")
    #     input_ids = encoded_input['input_ids']
    #     key = self._get_cache_key(input_ids)

    #     cached_output_sequences = self._check_cache(key)
    #     if cached_output_sequences:
    #         output_sequences = cached_output_sequences
    #     else:
    #         output_sequences = self.model.generate(input_ids=input_ids, attention_mask=encoded_input['attention_mask'], max_length=max_length)
    #         self._add_to_cache(key, output_sequences)
        
    #     return [self.tokenizer.decode(g, skip_special_tokens=True) for g in output_sequences]

    def generate(self, input_text, max_length=512):
        encoded_input = self.tokenizer(input_text, return_tensors="pt", max_length=max_length, truncation=True, padding="max_length")
        input_ids = encoded_input['input_ids']
        attention_mask = encoded_input['attention_mask']
        key = self._get_cache_key(input_ids)
    
        cached_output_sequences = self._check_cache(key)
        if cached_output_sequences is not None:
            output_sequences = torch.from_numpy(np.array(cached_output_sequences)).to(input_ids.device)
        else:
            output_sequences = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length)
            self._add_to_cache(key, output_sequences.cpu().numpy())
        
        return [self.tokenizer.decode(g, skip_special_tokens=True) for g in output_sequences]     
