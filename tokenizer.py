from collections import defaultdict, Counter
import regex as re
import json

class Tokenizer():
    def __init__(self, vocab_size: int, text: str = ""):
        self.vocab_size = vocab_size
        self.merges = {}
        self.eos_token = "<|endoftext|>" # The only special token in this model, the signal to end a sequence
        self.eos_token_id = self.vocab_size - 1

        # Andrej Karpathy's BPE tokenizer inspired by GPT4
        self.regex = re.compile(r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s""")
        
        self.word_freq = Counter()
        if text:
            self.set_training_text(text)

    def set_training_text(self, text: str):
        text_chunks = self.regex.findall(text)
        self.word_freq = Counter(tuple(chunk.encode("utf-8")) for chunk in text_chunks)

    def set_training_file(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            self.set_training_text(f.read())


    def _get_max_pair(self, word_freq: dict[tuple, int]):
        pairs = defaultdict(int)

        for word,freq in word_freq.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i+1])] += freq

        if not pairs:
            return None
        return max(pairs, key=pairs.get)

    def _merge(self, word_freq: dict[tuple, int], pair: tuple, new_id: int):
        new_word_freq = defaultdict(int)
        for word, freq in word_freq.items():   
            p0, p1 = pair
            if p0 not in word or p1 not in word:
                new_word_freq[word] = freq
                continue
            new_word = []
            i = 0

            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i+1]) == pair:
                    new_word.append(new_id)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_freq[tuple(new_word)] += freq
        return new_word_freq

            
        
    def _create_voc(self, word_freq: dict[tuple, int], vocab_size: int = 260):
        current_ind = 256
        current_word_freq = word_freq.copy()
        merges = {}

        while current_ind < vocab_size - 1:
            max_pair = self._get_max_pair(current_word_freq)
            if max_pair is None:
                break

            current_word_freq = self._merge(current_word_freq, max_pair, current_ind)
            merges[max_pair] = current_ind

            if current_ind % 100 == 0:
                print(f"Merge {current_ind}/{vocab_size - 1} : {max_pair} -> {current_ind}")

            current_ind += 1

        return merges

    def train(self):
        assert self.word_freq, "Aucun texte d'entraînement fourni."
        self.merges = self._create_voc(self.word_freq, self.vocab_size)

    def decode(self, ids: list[int]) -> str:
        inverted_merges = {v: k for k, v in self.merges.items()}
        eos_bytes = list(self.eos_token.encode("utf-8"))

        def decode_id(token_id: int) -> list[int]:
            if 0 <= token_id <= 255:
                return [token_id]
            elif token_id == self.eos_token_id:
                return eos_bytes
            elif token_id in inverted_merges:
                pair = inverted_merges[token_id]
                return decode_id(pair[0]) + decode_id(pair[1])
            else:
                return []

        byte_list = []
        for token_id in ids:
            byte_list.extend(decode_id(token_id))

        return bytes(byte_list).decode("utf-8", errors="replace")
    
    def _encode_chunk(self, chunk: list[int]) -> list[int]:
        if len(chunk) < 2:
            return chunk

        while len(chunk) >= 2:
            pairs = [(chunk[i], chunk[i + 1]) for i in range(len(chunk) - 1)]

            eligible_pairs = {pair: self.merges[pair] for pair in pairs if pair in self.merges}

            if not eligible_pairs:
                break 

            best_pair = min(eligible_pairs, key=eligible_pairs.get)
            new_id = eligible_pairs[best_pair]

            new_chunk = []
            i = 0
            while i < len(chunk):
                if i < len(chunk) - 1 and (chunk[i], chunk[i + 1]) == best_pair:
                    new_chunk.append(new_id)
                    i += 2
                else:
                    new_chunk.append(chunk[i])
                    i += 1
            chunk = new_chunk

        return chunk

    def encode(self, text: str) -> list[int]:
        text_chunks = self.regex.findall(text)
        
        encoded_tokens = []
        for chunk_str in text_chunks:
            chunk_bytes = list(chunk_str.encode("utf-8"))
            encoded_tokens.extend(self._encode_chunk(chunk_bytes))
            
        return encoded_tokens


    def save(self, file_path):
        assert self.merges
        merges_str = {f"{k[0]} {k[1]}" : v for k,v in self.merges.items()}

        config = {"vocab_size": self.vocab_size, "special_tokens": {self.eos_token : self.eos_token_id}, "merges" : merges_str}

        with open(file_path, "w", encoding='utf-8') as file:
            json.dump(config, file,ensure_ascii=False, indent=2)


    @classmethod
    def from_file(cls, file_path: str):
        with open(file_path, "r", encoding="utf-8") as file:
            config = json.load(file)

        tokenizer = cls(vocab_size=config["vocab_size"])
        
        special_tokens = config.get("special_tokens", {})
        if "<|endoftext|>" in special_tokens:
            tokenizer.eos_token = "<|endoftext|>"
            tokenizer.eos_token_id = special_tokens["<|endoftext|>"]

        tokenizer.merges = {
            tuple(map(int, k.split())): v 
            for k, v in config["merges"].items()
        }

        return tokenizer

