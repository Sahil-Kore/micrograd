import torch 
import torch.nn.functional as F
import matplotlib.pyplot as plt 

words=open('names.txt',"r").read().splitlines()

chars=sorted(list(set(''.join(words))))
stoi={ch:i for i,ch in enumerate(chars)}
stoi['.']=0

itos={v:k for k,v in stoi.items()}
vocab_size=len(itos)

block_size=3 # context size


def build_dataset(words):
    block_size=3
    X,Y=[],[]
    for w in words:
        
        context=[0] * block_size
        for ch in w + '.':
            ix=stoi[ch]
            X.append(context)
            Y.append(ix)
            context=context[1:] + [ix]

    X=torch.tensor(X)
    Y=torch.tensor(Y)
    print(X.shape,Y.shape)
    
    return X,Y

import random
random.seed(42)
random.shuffle(words)
n1=int(0.8 * len(words))
n2=int(0.9 * len(words))

Xtr,Ytr=build_dataset(words[:n1])
Xdev,Ydev=build_dataset(words[n1:n2])
Xte,Yte=build_dataset(words[n2:])

def cmp(s, dt, t ):
    ex=torch.all(dt == t.grad).item()
    app=torch.allclose(dt,t.grad)
    maxdiff = (dt - t.grad).abs().max().item()
    print(f'{s:5s} | exact: {str(ex):5s} | approximate: {str(app):5s} | maxdiff: {maxdiff} ')
    
n_embd = 10 # the dimensionality of the character embedding vectors
n_hidden = 64 # the number of neurons in the hidden layer of the MLP

g = torch.Generator().manual_seed(2147483647) # for reproducibility
C  = torch.randn((vocab_size, n_embd),            generator=g)
# Layer 1
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * (5/3)/((n_embd * block_size)**0.5)
b1 = torch.randn(n_hidden,                        generator=g) * 0.1 # using b1 just for fun, it's useless because of BN
# Layer 2
W2 = torch.randn((n_hidden, vocab_size),          generator=g) * 0.1
b2 = torch.randn(vocab_size,                      generator=g) * 0.1
# BatchNorm parameters
bngain = torch.randn((1, n_hidden))*0.1 + 1.0
bnbias = torch.randn((1, n_hidden))*0.1

# Note: I am initializating many of these parameters in non-standard ways
# because sometimes initializating with e.g. all zeros could mask an incorrect
# implementation of the backward pass.

parameters = [C, W1, b1, W2, b2, bngain, bnbias]
print(sum(p.nelement() for p in parameters)) # number of parameters in total
for p in parameters:
  p.requires_grad = True
batch_size=32
n=batch_size 

ix=torch.randint(0, Xtr.shape[0],(batch_size,),generator=g)
Xb,Yb=Xtr[ix],Ytr[ix]
# forward pass, "chunkated" into smaller steps that are possible to backward one at a time

emb = C[Xb] # embed the characters into vectors
embcat = emb.view(emb.shape[0], -1) # concatenate the vectors
# Linear layer 1
hprebn = embcat @ W1 + b1 # hidden layer pre-activation
# BatchNorm layer
bnmeani = 1/n*hprebn.sum(0, keepdim=True)
bndiff = hprebn - bnmeani
bndiff2 = bndiff**2
bnvar = 1/(n-1)*(bndiff2).sum(0, keepdim=True) # note: Bessel's correction (dividing by n-1, not n)
bnvar_inv = (bnvar + 1e-5)**-0.5
bnraw = bndiff * bnvar_inv
hpreact = bngain * bnraw + bnbias
# Non-linearity
h = torch.tanh(hpreact) # hidden layer
# Linear layer 2
logits = h @ W2 + b2 # output layer
# cross entropy loss (same as F.cross_entropy(logits, Yb))
logit_maxes = logits.max(1, keepdim=True).values
norm_logits = logits - logit_maxes # subtract max for numerical stability
counts = norm_logits.exp()
counts_sum = counts.sum(1, keepdims=True)
counts_sum_inv = counts_sum**-1 # if I use (1.0 / counts_sum) instead then I can't get backprop to be bit exact...
probs = counts * counts_sum_inv
logprobs = probs.log()
loss = -logprobs[range(n), Yb].mean()

# PyTorch backward pass
for p in parameters:
  p.grad = None
for t in [logprobs, probs, counts, counts_sum, counts_sum_inv, # afaik there is no cleaner way
          norm_logits, logit_maxes, logits, h, hpreact, bnraw,
         bnvar_inv, bnvar, bndiff2, bndiff, hprebn, bnmeani,
         embcat, emb]:
  t.retain_grad()
loss.backward()
loss

#calculating gradients manually

#log_probs is a tensor of size 32 which contains the probability of the correct index from each row  therefore the numbers which get picked will impact the loss others will not .
#the numbers whichh will get picked are then averaged so they will affect the loss by 1/n as they are averaged
#the numbers in the wrong indices will have gradient zero
#creating a tensor with zeros exaclty of the shape of log_probs
dlogprobs=torch.zeros_like(logprobs)
#set the correct indices gradient to 1/n
dlogprobs[range(n),Yb]=-1.0/n
cmp('logprobs',dlogprobs,logprobs)

dprobs=(1.0/probs) * dlogprobs
cmp('probs',dprobs,probs)

#probs= counts * counts_sum_inv
#counts 32x27
#counts_sum_inv 32x1
#counts * counts_sum_inv will multiply all the elements of the first row of couunts with the first row element of counts_sum_inv 
# for example 
'''
counts=a1 a2 a3
       b1 b2 b3
       c1 c2 c3

counts_sum_inv = s1
                 s2
                 s3

counts * counts_sum_inv=a1*s1 + a2*s1 +a3*s1
and so on

therefore every element of counts_sum_inv will affect probs not just one time but according to the numebr of columns and as the gradients get added up we will use sum  
''' 

#dprobs/dcounts_sum_inv = 
dcounts_sum_inv=(counts * dprobs).sum(1,keepdim=True)
cmp("dcounts_sum_inverse",dcounts_sum_inv,counts_sum_inv)

dcounts=counts_sum_inv * dprobs
 
dcounts_sum=(-counts_sum **-2) *dcounts_sum_inv

dcounts+=torch.ones_like(counts) *dcounts_sum
cmp("dcounts",dcounts,counts)

dnormlogits=counts *dcounts
cmp("normlogits",dnormlogits,norm_logits)

dlogits=dnormlogits.clone()
dlogit_maxes=(-dnormlogits).sum(1,keepdim=True)
cmp("dlogitmaxes",dlogit_maxes,logit_maxes)

dlogit_maxes
#the values of dlogit maxes are near zero because changing dlogitmaxes does not affect probs due to normalization 
dlogits+= F.one_hot(logits.max(1).indices,  num_classes=logits.shape[1]) * dlogit_maxes

cmp("logits", dlogits,logits)

dh=dlogits @ W2.T
dW2 = h.T @ dlogits
db2 = dlogits.sum(0)
cmp('h',dh,h)
cmp('W2',dW2,W2)
cmp('h',db2,b2)

dhpreact=(1.0 - h**2)* dh

cmp('hpreact',dhpreact,hpreact)
dbngain=(bnraw * dhpreact).sum(0,keepdim=True)
dbnraw=bngain * dhpreact
dbnbias = dhpreact.sum(0,keepdim=True)
cmp("bngain",dbngain,bngain)
cmp("dbnraw",dbnraw,bnraw)
cmp("dbnbias",dbnbias,bnbias)

dbndiff= bnvar_inv * dbnraw
dbnvar_inv= (bndiff * dbnraw).sum(0,keepdim=True)
cmp("dbnvar_inv",dbnvar_inv,bnvar_inv)

dbnvar= (-0.5 *(bnvar + 1e-5) ** -1.5)*dbnvar_inv

cmp("dbnvar", dbnvar,bnvar)

dbndiff2=(1.0/(n-1)) * torch.ones_like(bndiff2) * dbnvar
cmp('dbndiff2',dbndiff2,bndiff2)

dbndiff+=(2* bndiff) * dbndiff2
cmp("dbndiff",dbndiff,bndiff)

dhprebn = dbndiff.clone()
dbnmeani= (-dbndiff).sum(0)

dhprebn += 1.0/n * (torch.ones_like(hprebn) * dbnmeani)
cmp('hprebn',dhprebn,hprebn)

dembcat = dhprebn @ W1.T
dW1 = embcat.T @ dhprebn
db1 = dhprebn.sum(0)

cmp("dembcat",dembcat,embcat)
cmp("dW1",dW1,W1)
cmp("db1",db1,b1)

demb=dembcat.view(emb.shape)
cmp("demb",demb,emb)

dC = torch.zeros_like(C)
for k in range (Xb.shape[0]):
  for j in range (Xb.shape[1]):
    ix=Xb[k,j]
    dC[ix]+=demb[k,j]
    
cmp("C",dC,C)


#backward pass in a single step calculating dlogits


dlogits=F.softmax(logits,1)
dlogits[range(n),Yb] -=1
dlogits /=n
cmp('logits',dlogits,logits)

dhprebn =bngain*bnvar_inv/n * (n*dhpreact - dhpreact.sum(0)- n/(n-1) * bnraw*(dhpreact*bnraw).sum(0))
cmp("dhprebn",dhprebn,hprebn)



#full training loop without loss.backward()
# Exercise 4: putting it all together!
# Train the MLP neural net with your own backward pass

# init
n_embd = 10 # the dimensionality of the character embedding vectors
n_hidden = 200 # the number of neurons in the hidden layer of the MLP

g = torch.Generator().manual_seed(2147483647) # for reproducibility
C  = torch.randn((vocab_size, n_embd),            generator=g)
# Layer 1
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * (5/3)/((n_embd * block_size)**0.5)
b1 = torch.randn(n_hidden,                        generator=g) * 0.1
# Layer 2
W2 = torch.randn((n_hidden, vocab_size),          generator=g) * 0.1
b2 = torch.randn(vocab_size,                      generator=g) * 0.1
# BatchNorm parameters
bngain = torch.randn((1, n_hidden))*0.1 + 1.0
bnbias = torch.randn((1, n_hidden))*0.1

parameters = [C, W1, b1, W2, b2, bngain, bnbias]
print(sum(p.nelement() for p in parameters)) # number of parameters in total
for p in parameters:
  p.requires_grad = True

# same optimization as last time
max_steps = 200000
batch_size = 32
n = batch_size # convenience
lossi = []

# use this context manager for efficiency once your backward pass is written (TODO)
#with torch.no_grad():

# kick off optimization
for i in range(max_steps):

  # minibatch construct
  ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
  Xb, Yb = Xtr[ix], Ytr[ix] # batch X,Y

  # forward pass
  emb = C[Xb] # embed the characters into vectors
  embcat = emb.view(emb.shape[0], -1) # concatenate the vectors
  # Linear layer
  hprebn = embcat @ W1 + b1 # hidden layer pre-activation
  # BatchNorm layer
  # -------------------------------------------------------------
  bnmean = hprebn.mean(0, keepdim=True)
  bnvar = hprebn.var(0, keepdim=True, unbiased=True)
  bnvar_inv = (bnvar + 1e-5)**-0.5
  bnraw = (hprebn - bnmean) * bnvar_inv
  hpreact = bngain * bnraw + bnbias
  # -------------------------------------------------------------
  # Non-linearity
  h = torch.tanh(hpreact) # hidden layer
  logits = h @ W2 + b2 # output layer
  loss = F.cross_entropy(logits, Yb) # loss function

  # backward pass
  for p in parameters:
    p.grad = None
  loss.backward() # use this for correctness comparisons, delete it later!

  # manual backprop! #swole_doge_meme
  # -----------------
  # YOUR CODE HERE :)
  dC, dW1, db1, dW2, db2, dbngain, dbnbias  = None, None, None, None, None, None, None
  grads = [dC, dW1, db1, dW2, db2, dbngain, dbnbias]
  # -----------------

  # update
  lr = 0.1 if i < 100000 else 0.01 # step learning rate decay
  for p, grad in zip(parameters, grads):
    p.data += -lr * p.grad # old way of cheems doge (using PyTorch grad from .backward())
    #p.data += -lr * grad # new way of swole doge TODO: enable

  # track stats
  if i % 10000 == 0: # print every once in a while
    print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
  lossi.append(loss.log10().item())

  if i >= 100: # TODO: delete early breaking when you're ready to train the full net
    break