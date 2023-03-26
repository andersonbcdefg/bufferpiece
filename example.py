from datasets import load_dataset
from tokenizer import BufferPieceTokenizer

# train on bookcorpus
dataset = load_dataset("bookcorpus")
tokenizer = BufferPieceTokenizer()
tokenizer.train(32768, dataset['train'], model_prefix="bookcorpus")

# bilingual example
bilingual_tokenizer = BufferPieceTokenizer()
bilingual_tokenizer.train(2000, ["botchan.txt", "samples.txt"], model_prefix="bilingual")
ids = bilingual_tokenizer.encode_as_ids("哈囉世界! Hello, world! 🥺🥺🥺")
print(ids)
print(bilingual_tokenizer.decode(ids))