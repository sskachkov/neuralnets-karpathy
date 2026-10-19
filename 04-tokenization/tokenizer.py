import regex

class RegexTokenizer:
    BYTE_TO_CHAR = (
        "ĀāĂăĄąĆćĈĉĊċČčĎďĐđĒēĔĕĖėĘęĚěĜĝĞğĠ"          # bytes 0-32
        "!\"#$%&'()*+,-./0123456789:;<=>?@"           # bytes 33-64
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`"           # bytes 65-96
        "abcdefghijklmnopqrstuvwxyz{|}~"              # bytes 97-126
        "ġĢģĤĥĦħĨĩĪīĬĭĮįİıĲĳĴĵĶķĸĹĺĻļĽľĿŀŁł"          # bytes 127-160
        "¡¢£¤¥¦§¨©ª«¬"                                 # bytes 161-172
        "Ń"                                            # byte 173
        "®¯°±²³´µ¶·¸¹º»¼½¾¿"                          # bytes 174-191
        "ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖ×ØÙÚÛÜÝÞß"              # bytes 192-223
        "àáâãäåæçèéêëìíîïðñòóôõö÷øùúûüýþÿ"             # bytes 224-255
    )
    GPT4_SPLIT_PATTERN = regex.compile(r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""")

    def __init__(self, vocab_size, verbose=False):
        self.vocab_size = vocab_size
        self.verbose = verbose
        sorted_bytes = sorted(range(256), key=lambda b: ord(RegexTokenizer.BYTE_TO_CHAR[b]))
        self.vocab_id_to_byte = {i: b for i, b in enumerate(sorted_bytes)}
        self.byte_to_vocab_id = {b: i for i, b in enumerate(sorted_bytes)}
        self.merges = {}
        self.vocab = {}

    def train(self, text):
        iters = self.vocab_size - 256
        if self.verbose:
            print(f"training regex tokenizer (vocab_size={self.vocab_size}) with {iters} iterations...")
        strings = RegexTokenizer.GPT4_SPLIT_PATTERN.findall(text)
        list_of_bytes = [list(string.encode("utf-8")) for string in strings]

        list_of_ids = [[self.byte_to_vocab_id[b] for b in bytez] for bytez in list_of_bytes]
        idx = 256
        merges = {}

        vocab = {vocab_id : bytes([self.vocab_id_to_byte[vocab_id]]) for vocab_id in range(256)}
        for _ in range(iters):
            stats = self._get_stats(list_of_ids)
            if not stats:
                break
            most_common_pair = max(stats, key=stats.get)
            cnt = stats[most_common_pair]
            if cnt < 3:
                break
            list_of_ids = [self._merge(ids, most_common_pair, idx) for ids in list_of_ids]

            merges[most_common_pair] = idx
            vocab[idx] = vocab[most_common_pair[0]] + vocab[most_common_pair[1]]
            if self.verbose:
                print(f"{most_common_pair} [{vocab[most_common_pair[0]].decode(encoding='utf-8', errors='replace')}|{vocab[most_common_pair[1]].decode(encoding='utf-8', errors='replace')}] assigned to {idx} [{vocab[idx].decode(encoding='utf-8', errors='replace')}] with {stats[most_common_pair]} counts")
            idx += 1
        self.merges = merges
        self.vocab = vocab
        if self.verbose:
            print("training complete")

    def encode(self, text):
        strings = RegexTokenizer.GPT4_SPLIT_PATTERN.findall(text)
        list_of_ids = [[self.byte_to_vocab_id[b] for b in list(string.encode("utf-8"))] for string in strings]
        tokens = []
        for ids in list_of_ids:
            while len(ids) > 1:
                stats = self._get_stats([ids])
                pair = min(stats, key=lambda st:self.merges.get(st, float("inf")))
                if pair not in self.merges:
                    break
                idx = self.merges[pair]
                ids = self._merge(ids, pair, idx)
            tokens.extend(ids)
        return tokens

    def decode(self, ids):
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode(encoding="utf-8", errors="replace")
        return text

    def _get_stats(self, list_of_ids):
        d = {}
        for ids in list_of_ids:
            for t in zip(ids, ids[1:]):
                d[t] = d.get(t, 0) + 1
        return d

    def _merge(self, ids, pair, idx):
        ar = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                ar.append(idx)
                i += 2
            else:
                ar.append(ids[i])
                i += 1
        return ar

class Tokenizer:
    BYTE_TO_CHAR = (
        "ĀāĂăĄąĆćĈĉĊċČčĎďĐđĒēĔĕĖėĘęĚěĜĝĞğĠ"          # bytes 0-32
        "!\"#$%&'()*+,-./0123456789:;<=>?@"           # bytes 33-64
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`"           # bytes 65-96
        "abcdefghijklmnopqrstuvwxyz{|}~"              # bytes 97-126
        "ġĢģĤĥĦħĨĩĪīĬĭĮįİıĲĳĴĵĶķĸĹĺĻļĽľĿŀŁł"          # bytes 127-160
        "¡¢£¤¥¦§¨©ª«¬"                                 # bytes 161-172
        "Ń"                                            # byte 173
        "®¯°±²³´µ¶·¸¹º»¼½¾¿"                          # bytes 174-191
        "ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖ×ØÙÚÛÜÝÞß"              # bytes 192-223
        "àáâãäåæçèéêëìíîïðñòóôõö÷øùúûüýþÿ"             # bytes 224-255
    )    

    def __init__(self):
        sorted_bytes = sorted(range(256), key=lambda b: ord(Tokenizer.BYTE_TO_CHAR[b]))
        self.vocab_id_to_byte = {i: b for i, b in enumerate(sorted_bytes)}
        self.byte_to_vocab_id = {b: i for i, b in enumerate(sorted_bytes)}

    def _get_stats(self, tokens):
        d = {}
        for t in zip(tokens, tokens[1:]):
            d[t] = d.get(t, 0) + 1
        return d

    def _merge(self, ids, pair, idx):
        ar = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                ar.append(idx)
                i += 2
            else:
                ar.append(ids[i])
                i += 1
        return ar

    
    def train(self, text, vocab_size, verbose=False):
        iters = vocab_size - 256
        bytez = list(text.encode("utf-8"))
        tokens = [self.byte_to_vocab_id[b] for b in bytez]
        idx = 256
        merges = {}

        vocab = {vocab_id : bytes([self.vocab_id_to_byte[vocab_id]]) for vocab_id in range(256)}
        for _ in range(iters):
            stats = self._get_stats(tokens)
            most_common_pair = max(stats, key=stats.get)
            cnt = stats[most_common_pair]
            if cnt < 3:
                break
            tokens = self._merge(tokens, most_common_pair, idx)
            merges[most_common_pair] = idx
            vocab[idx] = vocab[most_common_pair[0]] + vocab[most_common_pair[1]]
            if verbose:
                print(f"{most_common_pair} [{vocab[most_common_pair[0]].decode(encoding='utf-8', errors='replace')}|{vocab[most_common_pair[1]].decode(encoding='utf-8', errors='replace')}] assigned to {idx} [{vocab[idx].decode(encoding='utf-8', errors='replace')}] with {stats[most_common_pair]} counts")
            idx += 1
        self.merges = merges
        self.vocab = vocab


    def encode(self, text):
        bytez = list(text.encode("utf-8"))
        tokens = [self.byte_to_vocab_id[b] for b in bytez]
        while len(tokens) > 1:
            stats = self._get_stats(tokens)
            pair = min(stats, key=lambda st:self.merges.get(st, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            tokens = self._merge(tokens, pair, idx)
        return tokens

    def decode(self, ids):
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode(encoding="utf-8", errors="replace")
        return text

def main():
    dataset_name = "input"
    text= """First Citizen:
    First, you know Caius Marcius is chief enemy to the people.
    """
    with open(f"datasets/{dataset_name}.txt", "r", encoding="utf-8") as f:
        training_text = f.read()
    print(len(training_text))

    vocab_size = 1024

    tokenizer = Tokenizer()
    tokenizer.train(training_text, vocab_size, verbose=True)

    reg_tokenizer = RegexTokenizer(vocab_size, verbose=True)
    reg_tokenizer.train(training_text)


    tok_tokens = tokenizer.encode(text)
    print(tok_tokens)
    print(", ".join([f"{token} [{tokenizer.vocab[token].decode("utf-8")}]" for token in tok_tokens]))
    tok_decoded = tokenizer.decode(tok_tokens)
    print(tok_decoded)
    print(f"tokenizer decoded text matches original: {tok_decoded == text}")

    reg_tok_tokens = reg_tokenizer.encode(text)
    print(reg_tok_tokens)
    print(", ".join([f"{token} [{reg_tokenizer.vocab[token].decode("utf-8")}]" for token in reg_tok_tokens]))

    reg_tok_decoded = reg_tokenizer.decode(reg_tok_tokens)
    print(reg_tok_decoded)
    print(f"regex tokenizer decoded text matches original: {reg_tok_decoded == text}")

if __name__ == "__main__":
    main()