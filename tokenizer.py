"""
Implementation of the BufferPiece tokenizer.
"""
class BufferPieceTokenizer:
  def __init__(self):
    self.pretokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    self.normalizer = tokenizers.normalizers.NFC()
    self.bpe_decoder = tokenizers.decoders.ByteLevel()
    self.model = None

  def pretokenize(self, s):
    normalized = self.normalizer.normalize_str(s)
    as_bytes = self.pretokenizer.pre_tokenize_str(normalized)[0][0]
    return as_bytes.replace("Ġ", " ")

  # Either accepts string or list of strings, but should only be 1 sequence (will be mushed together)
  def unpretokenize(self, s):
    if isinstance(s, str):
      return self.bpe_decoder.decode([s.replace(" ", "Ġ")])
    elif isinstance(s, list) or isinstance(s, np.ndarray):
      return self.bpe_decoder.decode(s)

  def load_from_file(self, tokenizer_path):
    self.model = spm.SentencePieceProcessor()
    self.model.load(tokenizer_path)
    self.unk_id = self.model.unk_id
    self.pad_id = self.model.pad_id
    self.bos_id = self.model.bos_id
    self.eos_id = self.model.eos_id

  def train(self, vocab_size, train_files, model_prefix, model_type="unigram", 
            span_length=500, special_tokens=[], temp_file_name="corpus.tmp",
            split_by_number=True, split_digits=True):
    combined_corpus = []
    print("Reading files into corpus...")
    if isinstance(train_files, datasets.Dataset):
      for idx, example in enumerate(train_files):
        file_text = example['text']
        as_bytes = self.pretokenize(file_text)
        chunks = [as_bytes[i:i+span_length] for i in range(0, len(as_bytes), span_length)]
        combined_corpus.extend(chunks)
        if idx > 1500000:
          break
    else:
      for file_path in train_files:
        with open(file_path) as f:
          file_text = f.read()
          as_bytes = self.pretokenize(file_text)
          chunks = [as_bytes[i:i+span_length] for i in range(0, len(as_bytes), span_length)]
          combined_corpus.extend(chunks)
    with open(temp_file_name, "w+") as f:
      lines_to_write = len(combined_corpus)
      print(f"Writing {lines_to_write} text spans to {temp_file_name}")
      for line in combined_corpus:
        f.write(line + "\n")
    cmd_string = f"--input={temp_file_name} --model_prefix={model_prefix} --vocab_size={vocab_size} --character_coverage=1.0"
    cmd_string += f" --max_sentence_length={span_length * 2} --normalization_rule_name=identity --add_dummy_prefix=false"
    cmd_string += f' --required_chars="{ALL_UNICODE_BYTES}"'
    if not split_by_number:
      cmd_string += " --split_by_number=false"
    if split_digits:
      cmd_string += " --split_digits=true"
    spm.SentencePieceTrainer.train(cmd_string)
    print(f"Saved model to {model_prefix}.model and vocab to {model_prefix}.vocab")
    self.load_from_file(f"{model_prefix}.model")

  def encode_as_ids(self, inputs):
    if self.model is None:
      raise RuntimeError("Cannot tokenize until tokenizer is trained or loaded from file.")
    if isinstance(inputs, str):
      return self.model.encode_as_ids(self.pretokenize(inputs))
    else:
      return [self.model.encode_as_ids(self.pretokenize(inp)) for inp in inputs]

  # Can only accept one list of ids
  def decode(self, ids):
    if self.model is None:
      raise RuntimeError("Cannot decode until tokenizer is trained or loaded from file.")
    out_bytes = self.model.decode_ids(ids)
    return self.unpretokenize(out_bytes)

  def batch_decode(self, batch):
    return [self.decode(x) for x in batch]

## TODO: Add tests
## TODO: Bring into line with the HF tokenizer interface, below:
"""
__init__(self, vocab_file: str, do_lower_case: bool = True, do_basic_tokenize: bool = True, never_split: Optional[List[str]] = None, unk_token: str = '[UNK]', sep_token: str = '[SEP]', pad_token: str = '[PAD]', cls_token: str = '[CLS]', mask_token: str = '[MASK]', tokenize_chinese_chars: bool = True, strip_accents=None, **kwargs): Initializes the tokenizer from a vocabulary file.

__call__(self, *args, **kwargs): Alias for self.encode().

convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]: Converts a token or a sequence of tokens into a single integer id or a sequence of ids using the vocabulary.

convert_ids_to_tokens(self, ids: Union[int, List[int]]) -> Union[str, List[str]]: Converts a single integer id or a sequence of ids into a token or a sequence of tokens using the vocabulary.

encode(self, text: Union[str, List[str]], text_pair: Optional[Union[str, List[str]]] = None, add_special_tokens: bool = True, padding: Union[bool, str] = False, truncation: Union[bool, str] = False, max_length: Optional[int] = None, stride: int = 0, return_tensors: Optional[Union[str, TensorType]] = None, **kwargs) -> Encoding: Encodes a text or a pair of texts into a sequence of integers using the tokenizer’s vocabulary.

decode(self, token_ids: Union[int, List[int]], skip_special_tokens: bool = False) -> str: Decodes a sequence of integers into a text string using the tokenizer’s vocabulary.

batch_encode_plus(self, *args, **kwargs): Encodes several sequences of text at once and returns them as a dictionary of arrays.

batch_decode(self, *args, **kwargs): Decodes several sequences of integers at once and returns them as a list of strings.
"""