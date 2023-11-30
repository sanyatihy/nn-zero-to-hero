import torch
import torch.nn as nn
import torch.nn.functional as F


# load data
with open('names.txt', 'r') as file:
    words = file.read().splitlines()

# create dictionary
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

# create the dataset
xs, ys = [], []
for w in words:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    xs.append(ix1)
    ys.append(ix2)
xs = torch.tensor(xs)
ys = torch.tensor(ys) # shifted
num = xs.nelement()
print('number of examples: ', num)

# initialize the 'network'
g = torch.Generator().manual_seed(2147483647)
embedding = nn.Embedding(num_embeddings=27, embedding_dim=1)
embedding.weight.data = torch.randn((27, 27), generator=g, requires_grad=True)

# gradient descent
for k in range(100):
  
  # forward pass
  logits = embedding(xs)
  loss = F.cross_entropy(logits, ys, reduction='mean')
  print(loss.item())
  
  # backward pass
  embedding.weight.grad = None # set to zero the gradient
  loss.backward()
  
  # update
  embedding.weight.data += -50 * embedding.weight.grad

# finally, sample from the 'neural net' model
g = torch.Generator().manual_seed(2147483647)

for i in range(10):
  
  out = []
  ix = 0
  while True:
    xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
    logits = xenc @ embedding.weight # predict log-counts
    p = F.softmax(logits, dim=1)
    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    out.append(itos[ix])
    if ix == 0:
      break

  print(''.join(out))
