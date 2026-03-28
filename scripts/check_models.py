# %% check rubert-tiny2 max length
from transformers import AutoConfig, AutoTokenizer

MODEL_NAME = "cointegrated/rubert-tiny2"

cfg = AutoConfig.from_pretrained(MODEL_NAME)
tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

print("max_position_embeddings:", getattr(cfg, "max_position_embeddings", None))
print("tokenizer.model_max_length:", tok.model_max_length)
print("tokenizer.max_len_single_sentence:", getattr(tok, "max_len_single_sentence", None))
