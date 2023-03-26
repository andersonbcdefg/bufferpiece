# BufferPiece
SentencePiece tokenizer that operates on UTF-8 bytes instead of Unicode characters.

## Motivation
The most common subword tokenizers used in 2023 for language modeling are byte-level BPE (used in GPT-2, GPT-3, RoBERTa) and SentencePiece (usually with Unigram LM model, used in T5, PaLM, and LLaMA). Each has benefits and drawbacks. My goal in this project is to develop a tokenizer with the benefits of both SentencePiece and byte-level BPE approaches, and the drawbacks of neither.

Byte-level BPE is a subword tokenization algorithm developed in 2015. It represents all words as sequences of bytes, and then iteratively merges the pairs of tokens that occur most frequently to create a subword vocabulary capable of representing any byte sequence. This elegantly avoids the problem of out-of-vocabulary tokens that occurs when using a fixed set of words as the vocabulary. It's also more efficient at compressing sequences compared to character- or byte-level tokenizers, which can represent any sequence, but lead to long sequences (since each token is one character), which are inefficient when fed to transformer models. Byte-level BPE also doesn't have to reserve tokens in the vocabulary for rare characters (e.g. in Chinese), because they can be represented as a sequence of multiple byte-level tokens.

Byte-level BPE has some shortcomings, though—at each step when learning the vocabulary, to decide what tokens to merge, BPE has to count all pairs of adjacent tokens to see which co-occur the most. This is made more tractable with *pre-tokenization*, which first splits the corpus into words and their frequencies. Then, you can count pairs of adjacent tokens for each word once, instead of for every time it occurs in the vocabulary—this makes it much more efficient. However, this does not work with languages like Chinese that do not use whitespace to separate words.

SentencePiece is a subword tokenizer developed in 2018 that works without whitespace pre-tokenization, which addresses this shortcoming of BPE. SentencePiece training can use either BPE or Unigram LM to construct a subword vocabulary. Its implementation of BPE is $O(N log(N))$ in the length of the training corpus, while Unigram LM is $O(N)$. These efficiency gains allow SentencePiece to be trained on entire sentences or paragraphs, rather than pre-tokenized words. This means it works out of the box with languages like Japanese or Chinese that don't put spaces between each word. It also supports subword regularization, which can make language modeling and translation more robust. 

However, SentencePiece has shortcomings of its own. For one thing, it operates on Unicode characters, rather than UTF-8 bytes, which means reserving tokens in vocabulary for rare characters, so that no character is "unknown." This doesn't matter for languages with a small number of characters, like English, but it's very relevant for, e.g., Chinese. SentencePiece addresses this by *falling back* on bytes for rare characters, but this leaves the decision of what fraction of characters to include in the vocabulary vs. fall back on bytes to the user. If a large fraction of characters (e.g. 0.9995, the default) are to be included in the vocabulary, then a big piece of the vocabulary is taken up by these characters, rather than larger units. SentencePiece is also not designed to work well with new line characters, as it's trained on a text file of sentences separated by new lines (and hence that do not contain new lines). The authors suggest [the _ad hoc_ workaround](https://github.com/google/sentencepiece/issues/101) of adding a special new-line character to the vocabulary and substituting that in (which is what the [GLM-130B paper](https://openreview.net/pdf?id=-Aw0rrrPUF) did), but it's clear that it's not designed to handle pieces of text that contains new line characters. This is a shortcoming when modeling or translating long documents, for which new lines are semantically important, separating paragraphs, sections, etc. that may be of different topics.

## BufferPiece Design
BufferPiece is a simple solution designed to bridge these two paradigms, keeping the strengths of both approaches. The algorithm is simple: we train a SentencePiece tokenizer on sequences of UTF-8 bytes, rather than Unicode characters. All input sequences (which may be sentences, paragraphs, or even longer spans of text) are pre-tokenized by converting them to bytes first, preserving spaces (if applicable) so that SentencePiece can use them to separate words. These byte sequences ("buffers") only have 256 possible UTF-8 "characters", so we can automatically guarantee full coverage, without reserving any vocabulary for rare characters. 

Buffers can contain new lines, rare characters in Asian scripts, and anything else—the byte-level encoding abstracts all these complications away. But since we're using SentencePiece, no whitespace pre-tokenization is required, so it doesn't matter whether the original span of text had whitespace or not. (If there *were* ordinary spaces in the original text, BufferPiece will still split on these to avoid weird situations like '\_and\_' and 'in\_the\' being tokens.)

In order to train the tokenizer to recognize and handle new lines, BufferPiece should be trained on spans of text that are not necessarily contained by one sentence or paragraph. Instead, split a document into chunks, and allow those chunks to cross sentence/paragraph boundaries. This is what natural text is like, and the tokenizer should be trained accordingly!

## Code & Experiments (Work in Progress)
All the work I've done on this so far was in a Colab notebook, where I tested the basic proof of concept, made sure it could train, and properly encode and decode text. I've started moving the code into this repository but it's still rough around the edges, I need to clean it up and write real tests.

After that, the plan is to test BufferPiece in the setting where it ought to be most advantageous—tokenization for Chinese, and bilingual tokenizers for machine translation between English and Chinese. The simplest approach is to train it on Chinese and English Wikipedia, and compare it to conventional tokenizers trained on the same corpus.

Right away, we can compare the tokenizers by how well they compress the languages that they're trained on. More important is the downstream performance, which I will measure on language modeling in English, language modeling in Chinese, and machine translation between English and Chinese.