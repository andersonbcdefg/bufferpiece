"""
Adapted from Andrej Karpathy's min-gpt BPE tokenizer. Used to get collection 
of byte-representations for the 256 bytes in the base 'vocab' of the tokenizer.
# https://github.com/karpathy/minGPT/blob/master/mingpt/bpe.py
"""
def byte_vocab():
    # the 188 chars that render fine in their original form and need no shifting
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:] 
    n = 0
    for b in range(2**8):
        if b not in bs:
            # if this byte is "ugly" then map it to the next available "nice" character
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    d = dict(zip(bs, cs))
    return "".join(d.values()).replace('"', "\\" + '"').replace("Ġ", "")