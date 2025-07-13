import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
words=open("names.txt","r").read().splitlines()

words[:3]
len(words)
chars=sorted(list(set(''.join(words))))
chars

stoi={k:i+1 for i,k in enumerate(chars)}
stoi['.']=0
stoi
itos={v:k for k,v in stoi.items()}
itos


#building the dataset
block_size=3  #context length:the number of letters used to predict the next one
X,Y=[],[]
for w in (words[:5]):
    
    print(w)
    context=[0] * block_size
    for ch in w + '.':
        ix=stoi[ch]
        X.append(context)
        Y.append(ix)
        print(''.join( itos[i] for i in context),"------>",itos[ix])
        context=context[1:] +[ix]  #crop and append

X=torch.tensor(X)
Y=torch.tensor(Y)

X.shape
Y.shape

#creating a lookup matrix C that contains the embedding for the 27 characters

C=torch.randn([27,2])
C

#every row of the matrix C is a embedding 
#instead of using one hot where the number of columns would be 27 we arer embedding in a 2d space so number of columns are 2

# ex :-embedding of d (stoi['d']=4) would be C[4]

C[5].shape

#we can also index on using a list 
#for example if we want to embeddings for a b c
#C[[1,2,3]]

C[[1,2,3]]

#we can also index using a matrix
#X is of size 32,3 so we it will treat every single element as an index and add its embedding in the third dimension
C[X].shape

#the 13 row and 2 column element of X
#this will return a tensor(1)
X[13][2]

#wwe can check that if we put  X[13,2]  as index which returns a 1 or explicitly index on 1 we get the same result
C[X[13,2]]
C[1]

#so we can get embeddngs for all the elements in X simultaneously
emb=C[X]


#creating the neural network

#100 neurons that take input of size 6
#input is 6 because we use context length 3 that is three words and each word has a two dimesional vector representation so we can think of it as embedding for sequence abc would be
#[[1,2],[34,12],[22,54]] that is we concatenated the embedings to get a vector for the sequence
#so we expanded the 2 d space to a 2x3 6d space 
W1=torch.randn((6,100))
b1=torch.randn(100)

#now we cannot multiply emb @ W1 because emb is a 3d matrix(32,3,2) and W1(32,6) is a 2d matrix
#we can imagine in emb that 32,3 character have their embeddings coming out of the page
#we want these to be squashed down so that the embeddings of lets say 0,0 element lay as a tuple and the embeddings for 0,1 is a tuple next to it  and 0,2 is next to it and all these tuples are inside a single tuple that is the first row of the matrix

#long story short flatten the matrix along the height 
#all the embeddings of the first element in first column
# that is if the first example was abc the first row of the matrix below is embeddingn for a 
emb[:,0,:]
#first row is embedding for b
emb[:,1,:]
#first row is embedding for c
emb[:,2,:]

#now we want to lay all these embeddings side by side along the columns so that we can get the embeddings for the whole exmaple

torch.cat([emb[:,0,:],emb[:,1,:],emb[:,2,:]],dim=1).shape
#the shape is 32,6 which means it can be multiplied with W1 which expects 6 inputs

#but this ,ethod is not flexible because if later change the dimension of embedding from 3 to say 5 this code will not work

#this works in the same manner it unbinds the 1 dimension and replaces all the contents in dimension 1 with list that contain the elements of dimension 2

torch.unbind(emb,1)
#no we need to concatenate it along the columns
torch.cat(torch.unbind(emb,1),1).shape

#this method is flexible 

# but there is a efficient way to do this 
#we can use the view function 
a=torch.arange(18)

#all these elements are aligned in the memomry as one after the other in a array
#we can use the view function to alter the way it is seen 
a.view(9,2)
a.view(2,9)
a.view(3,3,2)

#we can also think in reverse 
emb.view(32,6)

#this is better because this does not use new memory just changes the way we look at it

h=emb.view(32,6) @ W1 +b1
h.shape
#h contains the activation for these 32 examples for all the 100 neurons
#that is the output of all these 100 neurons 
#but hardcoding 32 is not good as later when we change the size of dataaset the number of examples will increase so instead we can use
h=emb.view(-1,6) @ W1 +b1

#if the second dimension is what would be the first dimension pytorch automatically calculates it as the number of elements in emb and emb.view should be same

#adding a activation
h=torch.tanh(emb.view(-1,6) @ W1 + b1)
h.shape
#27 neurons having 100 inputs
W2=torch.randn((100,27))
b2=torch.randn(27)

logits=h @ W2 +b2
logits.shape
counts=logits.exp()
prob=counts/counts.sum(1,keepdims=True)
prob.shape
Y

# Y contains the index for the correct output 
#now we want to pluck out the probabilities assigned by the neural network to the correct output

#the first row of prob contains the prob for the 27 characters to be the next character
#that is the prob of a being the next character is the row 1(example) column1(index of a) term
#if the first term of Y is 5 that means the 5th character is the correct answer so we check   the value of the  first row 5th column element which is the probability assigned by the neural and we want to maximize it

#look at all the rows and only the columns indexed by Y 
prob[torch.arange(32),Y] 

#cleaning and writing all of it at the same time
g=torch.Generator().manual_seed(2147483647)
C=torch.randn([27,2],generator=g)
W1=torch.randn((6,100),generator=g)
b1=torch.randn(100,generator=g)
W2=torch.randn((100,27),generator=g)
b2=torch.randn(27,generator=g)
parameters=[C,W1,b1,W2,b2]

sum(p.nelement() for p in parameters) #total number of parameters

# X is the integers correspding to the sequence of character X.shape(32,3) 32 examples 3 context length
emb=C[X] 
h=torch.tanh(emb.view(-1,6) @ W1 +b1) #(32,100)
logits= h @ W2 + b2

'''
earlier we did
counts=logits.exp()
prob=counts/counts.sum(1,keepdims=True)
loss=-prob[torch.arange(32),Y].log().mean() #nll loss

'''

loss=F.cross_entropy(logits,Y)
loss

#using cross entorpy loss is better because
'''
it does not create all the intermediate variables so efficient in terms of memory
backpropagation is easy and fast and earlier backpropagation had to go through all the intermediate node but cross entropy create a single node for output so backpropagation take only one step
also cross_entropy function is better numerically behaved 

ex if we have 100 as a count value
we then exponentiate it 
 e^100 which is a very large number we get inf as logit and probability as nan
 
how it does that
havign very high valued logits is problematic but very small does not cause a problem the count just nears 0
if we add/subtract  a constant value from   all the logits the probabilities dont change because of the averaging at the last step 
so cross_entropy finds the maximum value in logit array and subtracts the whole array by the maximum number so we dont run in inf issues 

gist
forward pass much more efficient
backward pass much nmore efficient 
numerically well behaved
'''
g=torch.Generator().manual_seed(2147483647)
C=torch.randn([27,2],generator=g)
W1=torch.randn((6,100),generator=g)
b1=torch.randn(100,generator=g)
W2=torch.randn((100,27),generator=g)
b2=torch.randn(27,generator=g)
parameters=[C,W1,b1,W2,b2]

#creating a training loop
for p in parameters:
    p.requires_grad=True

    
#the embeddings also change with gradient descent
for _ in range (1000):
    emb=C[X] 
    h=torch.tanh(emb.view(-1,6) @ W1 +b1) #(32,100)
    logits= h @ W2 + b2
    loss=F.cross_entropy(logits,Y)
    print(loss.item())
    #backward pass
    for p in parameters:
        p.grad=None
    loss.backward()
    
    #update
    for p in parameters:
        p.data+=(-0.1) * p.grad


logits.max(1)
#returns the max of each row 
#the max will be the prediction we can compare that with Y

#the loss is may not go to 0 because   
#for every example lets say emma the first input would be ...--->e that is ... should predict e
#for olivia ...  ----->o so ... must predict the first character of every name which is not possible 
#and as the input isze is only 32 we overfit on examples like oli,liv,emm ,mma

#now we want to train on the whole dataset

block_size=3  #context length:the number of letters used to predict the next one
X,Y=[],[]
for w in (words):
    
    context=[0] * block_size
    for ch in w + '.':
        ix=stoi[ch]
        X.append(context)
        Y.append(ix)
        context=context[1:] +[ix]  #crop and append

X=torch.tensor(X)
Y=torch.tensor(Y)

X.shape
Y.shape

emb=C[X]

g=torch.Generator().manual_seed(2147483647)
C=torch.randn([27,2],generator=g)
W1=torch.randn((6,100),generator=g)
b1=torch.randn(100,generator=g)
W2=torch.randn((100,27),generator=g)
b2=torch.randn(27,generator=g)
parameters=[C,W1,b1,W2,b2]

#creating a training loop
for p in parameters:
    p.requires_grad=True

    
#the embeddings also change with gradient descent
for _ in range (10):
    emb=C[X] 
    h=torch.tanh(emb.view(-1,6) @ W1 +b1) #(32,100)
    logits= h @ W2 + b2
    loss=F.cross_entropy(logits,Y)
    print(loss.item())
    #backward pass
    for p in parameters:
        p.grad=None
    loss.backward()
    
    #update
    for p in parameters:
        p.data+=(-0.1) * p.grad
#more 10 epochs

#training is slow as we are taking input of 228146 examples at time
#now training on minimbatches

torch.randint(0,X.shape[0],(32,))


#only work on 32 examples
for _ in range (100):
    #minibatch construct
    ix=torch.randint(0,X.shape[0],(32,))

    #forward pass
    emb=C[X[ix]] 
    h=torch.tanh(emb.view(-1,6) @ W1 +b1) #(32,100)
    logits= h @ W2 + b2
    loss=F.cross_entropy(logits,Y[ix])
    print(loss.item())
    #backward pass
    for p in parameters:
        p.grad=None
    loss.backward()
    
    #update
    for p in parameters:
        p.data+=(-0.1) * p.grad




#th gradient is not exact as we are not using the whole dataset but it is approximately right
#but backward pass is very very fast 
#it is much better to have a approximate gradient and take more steps 

#the loss being printed is only for the minibatch 
#so checking the loss for the entire dataset
emb=C[X] 
h=torch.tanh(emb.view(-1,6) @ W1 +b1) #(32,100)
logits= h @ W2 + b2
loss=F.cross_entropy(logits,Y)
print(loss)


#having a variable learning rate 

lre=torch.linspace(-3,1,1000)  #create eqaul sized 1000 intevals between -3 and 1 
lrs=10**lre   #learning rate goes from 0.001 to 1 in exponential scale


g=torch.Generator().manual_seed(2147483647)
C=torch.randn([27,2],generator=g)
W1=torch.randn((6,100),generator=g)
b1=torch.randn(100,generator=g)
W2=torch.randn((100,27),generator=g)
b2=torch.randn(27,generator=g)
parameters=[C,W1,b1,W2,b2]

#creating a training loop
for p in parameters:
    p.requires_grad=True

lri=[]
lossi=[]
    
#the embeddings also change with gradient descent
for i in range (1000):
    ix=torch.randint(0,X.shape[0],(32,))
    emb=C[X[ix]] 
    h=torch.tanh(emb.view(-1,6) @ W1 +b1) #(32,100)
    logits= h @ W2 + b2
    loss=F.cross_entropy(logits,Y[ix])
    print(loss.item())
    #backward pass
    for p in parameters:
        p.grad=None
    loss.backward()
    lr=lrs[i]
    #update
    for p in parameters:
        p.data+= -lr* p.grad
    
    #track stats
    lri.append(lre[i])
    lossi.append(loss.item())
    
plt.plot(lri,lossi)


# wwe see that that 10**-1 is a good learning rate to have so doing the trianing loop with constant lr




g=torch.Generator().manual_seed(2147483647)
C=torch.randn([27,2],generator=g)
W1=torch.randn((6,100),generator=g)
b1=torch.randn(100,generator=g)
W2=torch.randn((100,27),generator=g)
b2=torch.randn(27,generator=g)
parameters=[C,W1,b1,W2,b2]

#creating a training loop
for p in parameters:
    p.requires_grad=True

    
#the embeddings also change with gradient descent
for i in range (10000):
    ix=torch.randint(0,X.shape[0],(32,))
    emb=C[X[ix]] 
    h=torch.tanh(emb.view(-1,6) @ W1 +b1) #(32,100)
    logits= h @ W2 + b2
    loss=F.cross_entropy(logits,Y[ix])
    #backward pass
    for p in parameters:
        p.grad=None
    loss.backward()
    #update
    for p in parameters:
        p.data+= -0.1* p.grad
    
    
emb=C[X] 
h=torch.tanh(emb.view(-1,6) @ W1 +b1) #(32,100)
logits= h @ W2 + b2
loss=F.cross_entropy(logits,Y)
print(loss)


#leanrning rate decay reduce the lr by a factor of 10 when loss stops decreasing


for i in range (10000):
    ix=torch.randint(0,X.shape[0],(32,))
    emb=C[X[ix]] 
    h=torch.tanh(emb.view(-1,6) @ W1 +b1) #(32,100)
    logits= h @ W2 + b2
    loss=F.cross_entropy(logits,Y[ix])
    #backward pass
    for p in parameters:
        p.grad=None
    loss.backward()
    #update
    for p in parameters:
        p.data+= -0.01* p.grad
    
    
emb=C[X] 
h=torch.tanh(emb.view(-1,6) @ W1 +b1) #(32,100)
logits= h @ W2 + b2
loss=F.cross_entropy(logits,Y)
print(loss)


#train split-used to train the parameters
#val/dev split - used to tune hyper parameters
#test split -used to evaluate the model

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






g=torch.Generator().manual_seed(2147483647)
C=torch.randn([27,2],generator=g)
W1=torch.randn((6,300),generator=g)
b1=torch.randn(300,generator=g)
W2=torch.randn((300,27),generator=g)
b2=torch.randn(27,generator=g)
parameters=[C,W1,b1,W2,b2]
for p in parameters:
    p.requires_grad=True
    
lossi=[]
stepi=[]
for i in range (30000):
    ix=torch.randint(0,Xtr.shape[0],(32,))
    emb=C[Xtr[ix]] 
    h=torch.tanh(emb.view(-1,6) @ W1 +b1) #(32,100)
    logits= h @ W2 + b2
    loss=F.cross_entropy(logits,Ytr[ix])
    #backward pass
    for p in parameters:
        p.grad=None
    loss.backward()
    #update
    lr=0.01
    for p in parameters:
        p.data+= -lr* p.grad
    
    stepi.append(i)
    lossi.append(loss.item())

    
    
plt.plot(stepi,lossi)

emb=C[Xtr] 
h=torch.tanh(emb.view(-1,6) @ W1 +b1) #(32,100)
logits= h @ W2 + b2
loss=F.cross_entropy(logits,Ytr)
print(loss)




emb=C[Xdev] 
h=torch.tanh(emb.view(-1,6) @ W1 +b1) #(32,100)
logits= h @ W2 + b2
loss=F.cross_entropy(logits,Ydev)
print(loss)


#visualizing the word embeddings
plt.figure(figsize=(8,8))
plt.scatter(C[:,0].data,C[:,1].data,s=200)
for i in range(C.shape[0]):
    plt.text(C[i,0].item(),C[i,1].item(),itos[i],ha="center",va="center",color="white")
plt.grid("minor")




#increasing the dimensions for the vector space
#2->10

g=torch.Generator().manual_seed(2147483647)
C=torch.randn([27,10],generator=g)
W1=torch.randn((30,200),generator=g)
b1=torch.randn(200,generator=g)
W2=torch.randn((200,27),generator=g)
b2=torch.randn(27,generator=g)
parameters=[C,W1,b1,W2,b2]
for p in parameters:
    p.requires_grad=True
    
lossi=[]
stepi=[]

for i in range (50000):
    ix=torch.randint(0,Xtr.shape[0],(32,))
    emb=C[Xtr[ix]] 
    h=torch.tanh(emb.view(-1,30) @ W1 +b1) #(32,100)
    logits= h @ W2 + b2
    loss=F.cross_entropy(logits,Ytr[ix])
    #backward pass
    for p in parameters:
        p.grad=None
    loss.backward()
    #update
    lr=0.01
    for p in parameters:
        p.data+= -lr* p.grad
    
    stepi.append(i)
    lossi.append(loss.log10().item())

    
    
plt.plot(stepi,lossi)

emb=C[Xtr] 
h=torch.tanh(emb.view(-1,30) @ W1 +b1) #(32,100)
logits= h @ W2 + b2
loss=F.cross_entropy(logits,Ytr)
print(loss)




emb=C[Xdev] 
h=torch.tanh(emb.view(-1,30) @ W1 +b1) #(32,100)
logits= h @ W2 + b2
loss=F.cross_entropy(logits,Ydev)
print(loss)


#sampling from the model
for _ in range (20):
    out=[]
    context=[0] * block_size
    while True:
        emb=C[torch.tensor([context])]
        h=torch.tanh(emb.view(1,-1) @ W1 + b1 )
        logits=h @ W2 +b2
        probs=F.softmax(logits,dim=1)
        ix=torch.multinomial(probs,num_samples=1,generator=g).item()
        context=context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
    print(''.join(itos[i] for i in out))
