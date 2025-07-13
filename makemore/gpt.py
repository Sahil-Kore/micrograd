# import requests

# url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
# response = requests.get(url)

# # Save to file
# with open("tinyshakespeare.txt", "w", encoding='utf-8') as f:
#     f.write(response.text)

with open('./tinyshakespeare.txt',"r") as f:
    text=f.read()

len(text)

print(text[:1000])

chars=sorted(list(set(text)))
vocab_size=len(chars)
print(''.join(chars))
print(vocab_size)

stoi={ch:i for i,ch in enumerate(chars)}
itos={i:ch for ch,i in stoi.items()}
encode=lambda s:[stoi[c] for c in s] #encode a string output the integer
decode=lambda  l:''.join(itos[i] for i in l)  #decode a list of integers and return the string

print(encode("Hii there"))
print(decode([20, 47, 47, 1, 58, 46, 43, 56, 43]))

import torch
data=torch.tensor(encode(text) , dtype=torch.long)
data.shape,data.dtype
data[:1000]


n=int(len(data)*0.9)
train_data=data[:n]
val_data=data[n:]

block_size=8
train_data[:block_size+1]

#in a example of block size 8 there are 8 subexamples
#18, 47, 56, 57, 58,  1, 15, 47, 58]
#first example given 18 47 follows 
#2nd given 18 and 47 56 follows and so on

x=train_data[:block_size]
y=train_data[1:block_size+1]

for t in range( block_size):
    context=x[:t+1]
    target=y[t]
    print(f"When input is {context} out is {target}")

torch.manual_seed(1337)
batch_size=4 #how many independent sequences will be processed
block_size=8 #maximum conteext length for prediction

def get_batch (split):
    data=train_data if split=="train" else val_data
    ix=torch.randint(len(data) - block_size,(batch_size,))
    print(ix)
    x=torch.stack([data[i:i+block_size] for i in ix])
    y=torch.stack([data[i+1:i+block_size+1]for i in ix])
    return x,y

xb,yb =get_batch("train")
print("Inputs")
print(xb.shape)
print(xb)
print("targets")
print(yb.shape)
print(yb)

for b in range(batch_size):
    for t in range(block_size):
        context=xb[b,: t+1]
        target=yb[b,t]
        print(f"When input is {context} output is {target}")
        
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):
    
    def __init__(self,vocab_size):
        super().__init__()
        self.token_embedding_table=nn.Embedding(vocab_size,vocab_size)
        
    def forward(self,idx,targets=None):
        #idx and targets both are (B,T) shape tensors
        #initially the every character has their embedding added in the third dimension in logits
        #that is for xb cell there is vocab_size dimensional vector coming out of the page
        #this vocab_size vector will then act as logits 
        #so every position will be the probability of the next character 
        #the loss function will then be used to optimize the loss
        logits=self.token_embedding_table(idx)  #(B,T,C)

        #cross does requires the logits to be (B,C,T) or a 2d array of (T,C)
        #so to simply we just convert the logits to a 2d array same for target
        if targets is None:
            loss=None
        else:    
            B,T,C =logits.shape
            logits=logits.view(B*T,C)
            targets=targets.view(B*T)
            loss=F.cross_entropy(logits,targets)
        return logits,loss
    
    def generate(self,idx,max_new_tokens):
        #idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            #get the prediction
            logits,loss=self(idx)
            #focus only on the last time step
            logits=logits[: ,-1,:] #becomes (B,C)
            #apply softmax to get probabilites
            probs=F.softmax(logits,dim=-1) #(B,C)
            #sample from the distribution
            idx_next=torch.multinomial(probs,num_samples=1) #(B,1)
            #append the sampled index to the running sequence
            idx=torch.cat((idx,idx_next),dim=1) #(B,T+1)
        return idx

m=BigramLanguageModel(vocab_size)
logits,loss=m(xb,yb)

logits.shape
loss

idx=torch.zeros((1,1),dtype=torch.long)
idx
print(decode(m.generate(idx,max_new_tokens=100)[0].tolist()))


#training the model

optimizer=torch.optim.AdamW(m.parameters(),lr=1e-3)
batch_size=32
for steps in range(10000):
    #sample a batch of data
    xb,yb= get_batch('train')
    
    #evaulate the loss
    logits,loss=m(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())
print(decode(m.generate(idx,max_new_tokens=500)[0].tolist()))
