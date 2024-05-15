import importlib
from typing import Dict, Optional, Type
import torch.nn as nn
import redis
import pickle
from transformers import T5ForConditionalGeneration, T5Tokenizer
import numpy as np
import torch

class T5ForVLLMWithDistributedCache(torch.nn.Module):
    def __init__(self, model_dir, tokenizer_dir=None, cache_size=5000):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_dir)
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_dir if tokenizer_dir else model_dir)
        self.cache_size = cache_size
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        self.last_cache_hit = False  # Track the last cache hit status

    def _get_cache_key(self, input_ids):
        return str(input_ids.tolist())

    def _check_cache(self, key):
        cached_result = self.redis_client.get(key)
        if cached_result:
            self.last_cache_hit = True
            return pickle.loads(cached_result)
        self.last_cache_hit = False
        return None

    def _add_to_cache(self, key, value):
        self.redis_client.set(key, pickle.dumps(value))

    def generate(self, input_text, max_length=512):
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=max_length, truncation=True, padding="max_length")
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        key = self._get_cache_key(input_ids)

        output_sequences = self._check_cache(key)
        if output_sequences is None:
            output_sequences = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length)
            self._add_to_cache(key, output_sequences.cpu().numpy())
        
        return [self.tokenizer.decode(g, skip_special_tokens=True) for g in output_sequences], self.last_cache_hit

    def get_last_cache_hit(self):
        return self.last_cache_hit
