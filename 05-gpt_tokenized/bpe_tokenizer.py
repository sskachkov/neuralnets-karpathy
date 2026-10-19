from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import pickle
from typing import BinaryIO, Iterator

import regex


ENDOFTEXT_TOKEN = "<|endoftext|>"

RE_PRETOKEN = regex.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
RE_CUT = regex.compile(r"\s(?=\S)")

class BpeTrainer:
    def __init__(self, config, num_workers=4):
        self.vocab_size = config.vocab_size
        self.iterations = config.vocab_size - 256 - len(config.special_tokens)
        self.special_tokens = [t.encode("utf-8") for t in config.special_tokens]
        self.split_tokens = [t.encode("utf-8") for t in config.split_tokens]
        self.num_workers = num_workers
        pattern = b"|".join(regex.escape(t) for t in sorted(self.split_tokens, key=len, reverse=True)) or b"(?!)"
        self.re_docs_split = regex.compile(pattern)
        self.verbose = config.verbose
        self.dataset_name = config.dataset_name
        if self.verbose:
            print(f"Created BpeTrainer for {self.vocab_size} vocab size, {self.iterations} iterations.")

    def train_bpe(self):
        filepath = f"datasets/{self.dataset_name}.txt"
        with open(filepath, mode = "rb") as f:
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            f.seek(0)
            if file_size > 1024 ** 2:
                total = Counter()
                boundaries = self.find_chunk_boundaries(f, file_size)

                with ProcessPoolExecutor(max_workers=self.num_workers) as pool:
                    futures = [
                        pool.submit(self.count_shard, filepath, start, end) for start, end in boundaries
                    ]
                    if self.verbose:
                        print(f"Started {len(futures)} workers.")
                    for future in as_completed(futures):
                        partial = future.result()
                        total.update(partial)
            else:
                total = self.count_shard(filepath, 0, file_size)
        frequencies = {tuple(bytes([b]) for b in k.encode("utf-8")) : v for k,v in total.items()}

        vocab = {i : bytes([i]) for i in range(256)}
        if self.verbose:
            inverse_vocab = {v : k for k,v in vocab.items()}
        next_idx = len(vocab)
        merges = []

        words = [list(w) for w in frequencies]
        freqs = list(frequencies.values())
        pair_counts = Counter()
        pair_to_words = {}
        for i, (word,freq) in enumerate(frequencies.items()):
            for pair in zip(word, word[1:]):
                pair_counts[pair] += freq
                pair_to_words.setdefault(pair, set()).add(i)
        frequencies = None

        for i in range(self.iterations):
            if not pair_counts:
                break
            most_common = max(pair_counts, key=lambda tup: (pair_counts[tup], tup))
            occurrences = pair_counts[most_common]
            mcb = most_common[0] + most_common[1]
            merges.append(most_common)

            for wi in pair_to_words[most_common].copy():
                word = words[wi]
                w_freq = freqs[wi]
                new_word = []

                # remove
                for pair in zip(word, word[1:]):
                    pair_counts[pair] -= w_freq
                    if pair_counts[pair] == 0:
                        del pair_counts[pair]
                    pair_to_words[pair].discard(wi)

                # calc new word
                j = 0
                while j < len(word):
                    if j < len(word) - 1 and word[j] == most_common[0] and word[j+1] == most_common[1]:
                        new_word.append(mcb)
                        j+=2
                    else:
                        new_word.append(word[j])
                        j+=1
                words[wi] = new_word

                # update indicies with new_word structure
                for pair in zip(new_word, new_word[1:]):
                    pair_counts[pair] += w_freq
                    pair_to_words.setdefault(pair, set()).add(wi)
            del pair_to_words[most_common]
            vocab[next_idx] = mcb
            if self.verbose:
                inverse_vocab[mcb] = next_idx
                print(f"[{most_common[0].decode(encoding="utf-8", errors="replace")}] ({inverse_vocab[most_common[0]]}) | "
                      + f"[{most_common[1].decode(encoding="utf-8", errors="replace")}] ({inverse_vocab[most_common[1]]}) "
                      + f"-> [{mcb.decode(encoding="utf-8", errors="replace")}] ({inverse_vocab[mcb]}) "
                      + f"{occurrences} occurrences.")
            next_idx += 1

        # add special tokens
        for st in self.special_tokens:
            vocab[next_idx] = st
            if self.verbose:
                print(f"[{st.decode(encoding="utf-8", errors="replace")}] ({next_idx})")
            next_idx += 1
        return vocab, merges

    def tokenizer_path(self, dataset_name):
        return f"cache/bpe_tokenizer_{dataset_name}_vocab_{self.vocab_size}.pkl"

    def save(self, vocab, merges, special_tokens):
        if self.dataset_name is None:
            raise ValueError("No dataset name provided.")
        path = self.tokenizer_path(self.dataset_name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"vocab": vocab, "merges": merges, "special_tokens": special_tokens}, f)
        if self.verbose:
            print(f"Saved tokenizer to {path}.")

    def load(self):
        if self.dataset_name is None:
            raise ValueError("No dataset name provided.")
        path = self.tokenizer_path(self.dataset_name)
        with open(path, "rb") as f:
            data = pickle.load(f)
        if self.verbose:
            print(f"Loaded tokenizer from {path}.")
        return data["vocab"], data["merges"], data["special_tokens"]

    def load_or_train(self):
        if self.dataset_name is None:
            raise ValueError("No dataset name provided.")
        if os.path.exists(self.tokenizer_path(self.dataset_name)):
            return self.load()
        vocab, merges = self.train_bpe()
        special_tokens = [st.decode("utf-8") for st in self.special_tokens]
        self.save(vocab, merges, special_tokens)
        return vocab, merges, special_tokens

    def count_shard(self, filepath, start, end):
        with open(filepath, mode="rb") as f:
            f.seek(start)
            mini_chunk_size =10 * 1024 ** 2
            accum_chunk = b""
            counter = Counter()
            remaining = end - start
            while remaining:
                chunk = f.read(min(mini_chunk_size, remaining))
                if chunk == b"":
                    raise EOFError("Shard ends past EOF")
                remaining -= len(chunk)
                accum_chunk += chunk
                docs = self.re_docs_split.split(accum_chunk)
                for doc in docs[:-1]:
                    if doc:
                        text = doc.decode("utf-8", errors='replace')
                        pretokens = RE_PRETOKEN.findall(text)
                        counter.update(pretokens)
                accum_chunk = docs[-1]
            if accum_chunk:
                pretokens = RE_PRETOKEN.findall(accum_chunk.decode("utf-8", errors='replace'))
                counter.update(pretokens)
        return counter

    def find_chunk_boundaries(self, f : BinaryIO, file_size):
        chunk_size = file_size // self.num_workers
        chunk_boundaries = [i * chunk_size for i in range(self.num_workers + 1)]
        chunk_boundaries[-1] = file_size
        mini_chunk_size = 1024 ** 2
        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            f.seek(initial_position)
            while True:
                chunk = f.read(mini_chunk_size)
                if chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break
                match = self.re_docs_split.search(chunk)
                if match:
                    if initial_position + match.end() == file_size:
                        chunk_boundaries[bi] = file_size
                    else:
                        chunk_boundaries[bi] = initial_position + match.start()
                    break
                initial_position += mini_chunk_size

        chunk_boundaries = sorted(set(chunk_boundaries))
        zipped = list(zip(chunk_boundaries, chunk_boundaries[1:]))
        return zipped

class BpeTokenizer:
    def __init__(self, vocab, merges, special_tokens=None, verbose=False):
        self.vocab = vocab.copy()
        self.inverse_vocab = {v : k for k,v in vocab.items()}
        if not special_tokens:
            special_tokens = []
        self.special_tokens = special_tokens
        self.verbose = verbose
        for st in special_tokens:
            stb = st.encode("utf-8")
            if stb not in self.inverse_vocab:
                if verbose:
                    print("adding special token", stb)
                next_id = len(self.vocab)
                self.vocab[next_id] = stb
                self.inverse_vocab[stb] = next_id


        self.merges = [(self.inverse_vocab[m[0]], self.inverse_vocab[m[1]]) for m in merges]
        self.pair_to_rank = { pair : i  for i, pair in enumerate(self.merges)}
        self.pair_to_vocab_id = {pair : self.inverse_vocab[self.vocab[pair[0]] + self.vocab[pair[1]]] for pair in self.merges}
        if self.special_tokens:
            pattern = "(" + "|".join(regex.escape(t) for t in sorted(special_tokens, key=len, reverse=True)) + ")"
        else:
            pattern = "(?!)"
        self.re_special_tokens_split = regex.compile(pattern)
        self.max_special_len = max((len(t) for t in special_tokens), default=0)
        self.cache = {}

    def vocab_size(self):
        return len(self.vocab)

    def encode(self, text : str)-> list[int]:
        docs = self.re_special_tokens_split.split(text)
        tokens = []

        for doc in docs:
            if doc in self.special_tokens:
                tokens.append(self.inverse_vocab[doc.encode("utf-8")])
                continue

            pretokens = RE_PRETOKEN.findall(doc)
            for pretoken in pretokens:
                if pretoken in self.cache:
                    tokens.extend(self.cache[pretoken])
                    continue
                ids = [self.inverse_vocab[bytes([b])] for b in pretoken.encode("utf-8")]
                while len(ids) > 1:
                    pairs = {pair for pair in zip(ids, ids[1:])}
                    best_pair = min(pairs, key=lambda pair: self.pair_to_rank.get(pair, float("inf")))
                    if best_pair not in self.pair_to_rank:
                        break
                    idx = self.pair_to_vocab_id[best_pair]
                    # merge
                    new_ids = []
                    i = 0
                    while i < len(ids):
                        if i < len(ids) - 1 and ids[i] == best_pair[0] and ids[i+1] == best_pair[1]:
                            new_ids.append(idx)
                            i+=2
                        else:
                            new_ids.append(ids[i])
                            i+=1
                    ids = new_ids
                self.cache[pretoken] = ids
                tokens.extend(ids)
        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        hold = max(0, self.max_special_len - 1)
        buf = ""
        buf_len = 0
        for s in iterable:
            buf += s
            buf_len += len(s)
            if buf_len > 1024:
                match = self.re_special_tokens_split.search(buf)
                if match:
                    to_encode = buf[:match.end()]
                    ids = self.encode(to_encode)
                    yield from ids
                    buf = buf[match.end():]
                    buf_len = len(buf)
                    continue
            if buf_len > 16384:
                cut = None
                for m in RE_CUT.finditer(buf[:max(0, len(buf) - hold)]):
                    cut = m.start()
                if cut:
                    yield from self.encode(buf[:cut])
                    buf = buf[cut:]
                    buf_len = len(buf)
        if buf:
            ids = self.encode(buf)
            yield from ids


    def decode(self, ids: list[int]) -> str:
        bytez = b"".join([self.vocab[idx] for idx in ids])
        return bytez.decode("utf-8", errors="replace")

    def tokenizer_id(self):
        return 'bpe_tokenizer'