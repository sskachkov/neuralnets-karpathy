import torch
import os
from gpt_model import ModelConfig
import regex as re
import random

DATASET_CACHE_DIR = "cache"

class DataProvider:
    def __init__(self, config : ModelConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.split_documents_pattern = "|".join(f"(?<={re.escape(t)})" for t in self.config.split_tokens)
        self.prepare_dataset()

    def get_batch(self, data_type):
        split_data = self.get_data(data_type)
        block_size = self.config.block_size
        batch_size = self.config.batch_size
        
        ix = torch.randint(len(split_data) - block_size, (batch_size,))
        x = torch.stack([split_data[i:i+block_size] for i in ix])
        y = torch.stack([split_data[i+1:i+block_size+1] for i in ix])
        return x.to(self.config.device), y.to(self.config.device)

    def get_data(self, data_type):
        return self.train_data if data_type == 'train' else self.val_data

    def _load_dataset_text(self, dataset_name):
        with open(f"datasets/{dataset_name}.txt", 'r', encoding='utf-8') as f:
            text = f.read()
        return text

        
    def _cache_path(self):
        dataset_name = self.config.dataset_name
        return f"{DATASET_CACHE_DIR}/dataset_{dataset_name}_{self.tokenizer.tokenizer_id()}_vocab_{self.config.vocab_size}_val_{self.config.val_ratio}_seed_{self.config.seed}.pt"

    def prepare_dataset(self, cache=True):
        config = self.config
        dataset_name = self.config.dataset_name

        cache_path = self._cache_path() if dataset_name and cache else None
        if cache_path and os.path.exists(cache_path):
            if config.verbose:
                print(f"loading cached tokenized dataset from {cache_path}..")
            data = torch.load(cache_path)
            self.train_data, self.val_data, self.chars_per_token = data['train'], data['val'], data['cpt']
            
            return

        if not dataset_name:
            raise ValueError("prepare_dataset needs either text or a dataset_name to load it from")
        if config.verbose:
            print(f"no cache at {cache_path}, loading raw text for '{dataset_name}'..")
        text = self._load_dataset_text(dataset_name)
        documents = re.split(self.split_documents_pattern, text)
        n_val_docs = int(len(documents) * config.val_ratio)

        if n_val_docs >= config.min_val_docs:
            if config.verbose:
                print(f"found {len(documents)} separated documents ({n_val_docs} to val) — using document-boundary split")
            random.Random(config.seed).shuffle(documents)
            train_text = "".join(documents[n_val_docs:])
            val_text = "".join(documents[:n_val_docs])
            self.train_data = torch.tensor(self.tokenizer.encode(train_text), dtype=torch.long)
            self.val_data = torch.tensor(self.tokenizer.encode(val_text), dtype=torch.long)
        else:
            if config.verbose:
                print(f"only {len(documents)} separated documents found — not enough for a document-boundary split, falling back to position-based split")

            
            data = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
            chunk_size = max(1024,int(len(data) / (1.1 * config.min_val_docs / config.val_ratio)))
            chunks = data.split(chunk_size)

            g = torch.Generator().manual_seed(config.seed)
            perm = torch.randperm(len(chunks), generator=g)
            n_val = max(1, int(len(chunks) * config.val_ratio))
            val_ids = set(perm[:n_val].tolist())
            self.train_data = torch.cat([c for i, c in enumerate(chunks) if i not in val_ids])
            self.val_data = torch.cat([c for i, c in enumerate(chunks) if i in val_ids])

        self.chars_per_token = len(text) / (len(self.train_data) + len(self.val_data))

        if cache_path:
            os.makedirs(DATASET_CACHE_DIR, exist_ok=True)
            if config.verbose:
                print(f"caching tokenized dataset to {cache_path}..")
            data = {'train' : self.train_data, 'val' : self.val_data, 'cpt' : self.chars_per_token}
            torch.save(data, cache_path)
