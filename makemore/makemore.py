words=open('names.txt',"r").read().splitlines()

len(words)

words[:10]

min(len(w) for w in words)
max(len(w) for w in words)

#bigram model
#only working with two characters at a time 
#that is read a single character of a name 
#given the current character predict the next character


#maintains count for the bigrams
b={}
#pairing two characters at a time
for w in words:
    chs=['<S>'] + list(w) + ["<E>"]
    for ch1,ch2 in zip(chs,chs[1:]):
        bigram=(ch1,ch2)
        #add the count of bigram in dict if bigram is not in dict initialize with zero
        b[bigram]=b.get(bigram,0)+1
      
#adding the starting and ending special characters to tell the model later when to start and when to end

#sorted by default sorts items by the keys but we want to sort by values so using a lambda which takes in input a key value pair kv[1] is the value
#negative sign to sort in descending order
sorted(b.items(),key=lambda kv:-kv[1])

#we store the bigrams and their counts as a 2d tensor where the rows are the first character of bigram and colmns represent the second word of the bigram and value at the cell represent the count

#ex
#   n
#a  5438
  
import torch
#26 letters and end and start symbols
N=torch.zeros((28,28),dtype=torch.int32)

chars=sorted(list(set(''.join(words))))
chars

#string to integer  lookup table
stoi={s:i for i,s in enumerate(chars)}
stoi

stoi['<S>']=26
stoi["<E>"]=27

for w in words:
    chs=["<S>"] + list(w) + ["<E>"]
    for ch1,ch2 in zip(chs,chs[1:]):
        ix1=stoi[ch1]
        ix2=stoi[ch2]
        N[ix1,ix2] +=1
        
import matplotlib.pyplot as plt

plt.imshow(N)

itos={i:s for s,i in stoi.items()}
itos

plt.figure(figsize=(16,16))
plt.imshow(N,cmap="Blues")
for i in range (28):
    for j in range(28):
        chstr=itos[i] + itos[j]
        plt.text(j,i,chstr,ha="center",va="bottom",color="gray")
        plt.text(j,i,N[i,j].item(),ha="center",va="top",color="gray")
plt.axis("off")

#problems 
# we have a entire row zero where the <E> symbol is the staring symbol as the ending symbol can never occur before a letter in bigram the whole tow is zero
#similiarly a whole column is zero where <S> is the follow up character in a bigram

#this is waste of space

#also <s> and <e> are very crowded here and also they eat up size eual to three characters 

N=torch.zeros((27,27),dtype=torch.int32)

chars=sorted(list(set(''.join(words))))
chars

#replacing start and end symbol with dot
#offsetting all the characters by 1 so that the special symbol can have the first place (pleasing to see)

stoi={s:i+1 for i,s in enumerate(chars)}
stoi['.']=0
itos={i:s for s,i in stoi.items()}


for w in words:
    chs=["."] + list(w) + ["."]
    for ch1,ch2 in zip(chs,chs[1:]):
        ix1=stoi[ch1]
        ix2=stoi[ch2]
        N[ix1,ix2] +=1
        
import matplotlib.pyplot as plt

plt.figure(figsize=(16,16))
plt.imshow(N,cmap="Blues")
for i in range (27):
    for j in range(27):
        chstr=itos[i] + itos[j]
        plt.text(j,i,chstr,ha="center",va="bottom",color="gray")
        plt.text(j,i,N[i,j].item(),ha="center",va="top",color="gray")
plt.axis("off")

#looking at the first row
#the first row is the one with dot 
#so the rows has the frequency of other characters following dot
N[0,]

#converting frequencies to probability
p=N[0].float()
p=p/p.sum()
p

#p represents the probabilities of a character following dot which is the same as the probability to be the first character of the name

#now we can sample from this tensor 
#to keep it deterministic we are usign pytorch generator

g=torch.Generator().manual_seed(2147483647)
p=torch.rand(3,generator=g)
p=p/p.sum()
p
torch.multinomial(p,num_samples=100,replacement=True,generator=g)



#sampling from our ditribution which is the first row for now
p=N[0].float()
p=p/p.sum()

g=torch.Generator().manual_seed(2147483647)
ix=torch.multinomial(p,num_samples=1,replacement=True,generator=g).item()
ix
itos[ix]


#loop to get a sequnce of characters
g=torch.Generator().manual_seed(2147483647)
for i in range (20):
    ix=0
    out=[]
    while True:
        p=N[ix].float()
        p=p/p.sum()
        ix=torch.multinomial(p,num_samples=1,replacement=True,generator=g).item()
        out.append(itos[ix])
        if ix==0:
            break
    print(''.join(out))




#creating a matrix P which will contain the probabilities as before we convert rows to probabilities multiple times

P=N.float()
P.shape

#do the sum on the 0th dimension that is rows
#keep dim =True does not squueze the useless dimension

P.sum(1,keepdim=True)
#a column vector that contains the sum of that particular row

#according to broadcast rules 
#we can divide the rows P by the corresponding values in P.sum(1,keepdim=True)

#broadcasting rules
'''
align the two tensors dimensions to the right

start from the trailing dimensions that is read from right to left
the pair of dimension in the two tensors should be equal or the second one should be one or zero

what happens 
we have 27x27 and  27x1 matrices

so the 27x1 matrix is streched out to be 27x27 where every row has identical values corresponding to the entries in the column vector 27x1
'''

#using inplace operation because it does not create new memory and is faster
P/=P.sum(1,keepdim=True)
g=torch.Generator().manual_seed(2147483647)
for i in range (20):
    ix=0
    out=[]
    while True:
        p=P[ix]
        ix=torch.multinomial(p,num_samples=1,replacement=True,generator=g).item()
        out.append(itos[ix])
        if ix==0:
            break
    print(''.join(out))



'''
instead if just say
P.sum(1,keepdim=false)  {false by default}
we get a tensor of size 27
that is a one dimensional vector the extra dimension is squeezed

when performing division now
P=P/P.sum(1)

what happens under the hood
27 x 27 and 27
align the dimension to the right

27 x 27
     27

according to broadcasting rules in the last dimensions are equal passed the test
and the first dimensions one of them is non existent or zero so passed the test
so the operation is valid
but  it does not give the required result

internally the 27 one d tensor becomes a 1 x 27 row vector

27 x 27
1  x 27
so now we unsqueeze the row vector to make it a 27 x 27 matrix where every column has the same value as the correspondign entry in the row vector
which gives the wrong result 

this can be checked by summing the rows of the resultant and the resultant should be 1 as we are summing probabilities of any character following the given character but the answer is not one as the rows have been not normalized but the columnshave been

'''

P=P/P.sum(1)
P[0,:].sum()


#calculating loss

for w in words[:3]:
    chs=["."] + list(w) + ["."]
    for ch1 ,ch2 in zip(chs,chs[1:]):
        ix1=stoi[ch1]
        ix2=stoi[ch2]
        prob=P[ix1,ix2]
        print(f'{ch1}{ch2}: {prob:.4f}')
        
        
#the prob stat tell us the probability assigned by the model to these bigrams if it were randomly guessign the nxt letter then the probability would be 1/27 roughly 4 percent but we can see that most of them are above 1/27 so we can say the model has learned something

#ideally we want the probabilities to be near one and that would mean the model is correctly guessing the next character

#to calculate the loss we use likelihood  The probability that the model assigns to the entire dataset, based on its parameters.so
#The likelihood is the model’s estimate of how likely it is that the observed data came from this model.

#or how i see it ,it is the probability that the model guesses the whole training data correctly

#P(emma)*P(olivia)*....P(last name)

#product of all the probabilites of getting all the bigrams

#as the probabilites are very small we use log probabilities

#also log(a*b*c)=loga +logb +logc

log_likelihood=0.0
for w in words[:3]:
    chs=["."] + list(w) + ["."]
    for ch1 ,ch2 in zip(chs,chs[1:]):
        ix1=stoi[ch1]
        ix2=stoi[ch2]
        prob=P[ix1,ix2]
        logprob=torch.log(prob)
        log_likelihood+=logprob
        print(f'{ch1}{ch2}: {logprob:.4f}')
        
        
#we want the log_likelihood to be as large as possible
log_likelihood

#but loss functions are designed to be minimized and log_likelihood is supposed to be maximized

#so we use negative log likelihood

nll=-log_likelihood

nll

#and usually we use the average negative likelihood
        

log_likelihood=0.0
n=0
for w in words[:3]:
    chs=["."] + list(w) + ["."]
    for ch1 ,ch2 in zip(chs,chs[1:]):
        ix1=stoi[ch1]
        ix2=stoi[ch2]
        prob=P[ix1,ix2]
        logprob=torch.log(prob)
        log_likelihood+=logprob
        n+=1
        print(f'{ch1}{ch2}: {logprob:.4f}')
        
nll=-log_likelihood
print("average negative log likelihood",nll/n)

#now we want to minimize nll/n

'''
maximize likelihood product of probabilities
convert it to log likelihood and maximize it as log is monotonically increasing
which is equal to minimize negative log likelihood
and average it 
'''
#for whole dataset
log_likelihood=0.0
n=0
for w in words:
    chs=["."] + list(w) + ["."]
    for ch1 ,ch2 in zip(chs,chs[1:]):
        ix1=stoi[ch1]
        ix2=stoi[ch2]
        prob=P[ix1,ix2]
        logprob=torch.log(prob)
        log_likelihood+=logprob
        n+=1
        
nll=-log_likelihood
print("average negative log likelihood",nll/n)

#testing for any name
log_likelihood=0.0
n=0
for w in ["andrejq"]:
    chs=["."] + list(w) + ["."]
    for ch1 ,ch2 in zip(chs,chs[1:]):
        ix1=stoi[ch1]
        ix2=stoi[ch2]
        prob=P[ix1,ix2]
        logprob=torch.log(prob)
        log_likelihood+=logprob
        print(f"{ch1}{ch2}:{prob:.4f} {logprob:.4f}")
        n+=1
        
nll=-log_likelihood
print("average negative log likelihood",nll/n)


#we get infinite loss because the probability of getting jq as a bigram is 0 as j is followed by q is 0 times

#this is not desired so we do smoothing
#so we add 1 to count of every bigram 
#so the probability of getting any bigram is not 0 

P=(N+1).float()
P/=P.sum(1,keepdim=True)
P/=P.sum(1,keepdim=True)
g=torch.Generator().manual_seed(2147483647)
for i in range (20):
    ix=0
    out=[]
    while True:
        p=P[ix]
        ix=torch.multinomial(p,num_samples=1,replacement=True,generator=g).item()
        out.append(itos[ix])
        if ix==0:
            break
    print(''.join(out))

log_likelihood=0.0
n=0
for w in ["andrejq"]:
    chs=["."] + list(w) + ["."]
    for ch1 ,ch2 in zip(chs,chs[1:]):
        ix1=stoi[ch1]
        ix2=stoi[ch2]
        prob=P[ix1,ix2]
        logprob=torch.log(prob)
        log_likelihood+=logprob
        print(f"{ch1}{ch2}:{prob:.4f} {logprob:.4f}")
        n+=1
        
nll=-log_likelihood
print("average negative log likelihood",nll/n)

#now the nll is not infinity
