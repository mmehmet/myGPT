import tiktoken

enc = tiktoken.get_encoding('gpt2')
enc.n_vocab

print(enc.encode('hii there'))
