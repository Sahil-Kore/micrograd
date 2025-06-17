import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def f(x):
    return 3*x**2 - 4*x+5


f(3.0)

xs=np.arange(-5,5,0.25)
ys=f(xs)

plt.plot(xs,ys)
sns.lineplot(x=xs,y=ys)

# derivative lim h->0 f(a+h)-f(a)/h
#taking a very small value of h to demonstrate

h=0.0000001
x=3.0
f(x+h)
f(x+h)-f(x)
(f(x+h)-f(x))/h

#slope is negative


x=2/3
f(x+h)
f(x+h)-f(x)
(f(x+h)-f(x))/h


a=2.0
b=-3.0
c=10.0
d=a*b+c
print(d)


h=0.000001
d1=a*b+c
#derivative wrt a
a+=h
d2=a*b+c
print("d1 is ",d1)
print("d2 is ",d2)
print("Slope is ",(d2-d1)/h)



d1=a*b+c
#derivative wrt b
b+=h
d3=a*b+c
print("d1 is ",d1)
print("d3 is ",d3)
print("Slope is ",(d3-d1)/h)



d1=a*b+c
#derivative wrt c
c+=h
d4=a*b+c
print("d1 is ",d1)
print("d3 is ",d4)
print("Slope is ",(d4-d1)/h)


class Value:
    def __init__(self,data,_children=(),_op='',label=''):
        self.data=data
        self._prev=set(_children)
        self._op=_op
        self.label=label
        self.grad=0.0
        self._backward=lambda:None
          
    def __repr__(self):
        return f"Value(data={self.data})"    
    
    def __add__(self,other):
        other=other if isinstance(other,Value) else Value(other)
        out= Value(self.data + other.data,(self,other),'+')

        def _backward():
            self.grad+=1.0 * out.grad
            other.grad+=1.0 * out.grad
        
        out._backward=_backward
        return out    

    def __radd__(self,other):
        return self+other
    def __mul__(self,other):
        other=other if isinstance(other,Value) else Value(other)
        out= Value(self.data * other.data,(self,other),'*')

        def _backward():
            self.grad+= other.data * out.grad
            other.grad+=self.data * out.grad
        
        out._backward=_backward
        return out
    
    def exp(self):
        x=self.data
        out=Value(math.exp(x),(self,),'exp')
        
        def _backward():
            self.grad+=out.data * out.grad
        
        out._backward=_backward
        return out
    
    #we can write a/b as 
    #a* b**-1
    
    def __pow__(self,other):
        assert isinstance(other,(int,float))
        out=Value(self.data**other,(self,),f"**{other}")

        def _backward():
            self.grad+= other*(self.data**(other-1))*out.grad
        
        out._backward=_backward
        return out
    
    def __truediv__(self,other):
        return self * other**-1
    
    #Value* int is works
    #but int * value doesnt because int will call int.__mul__() which is not defined for value so we use rmul to see if value * int is defined or not
    def __rmul__(self,other):
         return self*other
    
    
    def tanh(self):
        x=self.data
        t=(math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out=Value(t,(self,),"tanh")

        def _backward():
            self.grad+=(1-t**2) * out.grad
        
        out._backward=_backward
        return out
    
    def backward(self):
        
        topo=[]
        visited=set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad=1.0
        for node in reversed (topo):
            node._backward()

    
    def __neg__(self):
        return self*-1
    
    def __sub__(self,other):
        return self + (-other)

a=Value(2.0,label='a')
print(a)

b=Value(-3.0,label='b')
a+b

c=Value(10.0,label='c')
e=a*b
e.label='e'

d=e+c
d.label='d'

f=Value(-2.0,label='f')
L=d*f
L.label="L"
L

from graphviz import Digraph

def trace(root):
  # builds a set of all nodes and edges in a graph
  nodes, edges = set(), set()
  def build(v):
    if v not in nodes:
      nodes.add(v)
      for child in v._prev:
        edges.add((child, v))
        build(child)
  build(root)
  return nodes, edges

def draw_dot(root):
  dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right
  
  nodes, edges = trace(root)
  for n in nodes:
    uid = str(id(n))
    # for any value in the graph, create a rectangular ('record') node for it
    dot.node(name = uid, label = "{ %s | data %.4f |grad %.4f }" % (n.label, n.data,n.grad), shape='record')
    if n._op:
      # if this value is a result of some operation, create an op node for it
      dot.node(name = uid + n._op, label = n._op)
      # and connect this node to it
      dot.edge(uid + n._op, uid)

  for n1, n2 in edges:
    # connect n1 to the op node of n2
    dot.edge(str(id(n1)), str(id(n2)) + n2._op)

  return dot

draw_dot(L)

def lol():
    a=Value(2.0,label='a')
    b=Value(-3.0,label='b')
    c=Value(10.0,label='c')
    e=a*b
    e.label='e'
    d=e+c
    d.label='d'
    f=Value(-2.0,label='f')
    L=d*f
    L.label="L"
    L1=L.data
    
    #derivative wrt to a
    a=Value(2.0+h,label='a')
    b=Value(-3.0,label='b')
    c=Value(10.0,label='c')
    e=a*b
    e.label='e'
    d=e+c
    d.label='d'
    f=Value(-2.0,label='f')
    L=d*f
    L.label="L"
    L2=L.data
    
    print((L2-L1)/h)
    
lol()

#derivative of l with itself should be 1
L.grad=1
#manually setting some gradient
f.grad=4.0
d.grad=-2.


#for plus nodes it just copies the derivative of its parent note root is the rightmost node

'''
dl/de=dl/dd * dd/de

since d=e+c
dl/de=1
therefore
dl/de=dl/dd

plus distributes the gradient to its children
'''

c.grad=-2.0
e.grad=-2.0


'''
dl/da=dl/de * de/da
dl/de= -2.0     calculated earlier

e=a*b
de/da=b

b=-3.0

dl/da=-3.0 * -2.0
dl/da=6

similiarly 
dl/db= 2.0 * -2.0
dl/db=-4.0
'''

a.grad=6.0
b.grad=-4.0


#changing the inputs and observing the amount of change in result L

#if we go in the direction of gradient we will increase the result 
a.data+=0.01 * a.grad
b.data+=0.01 * b.grad
c.data+=0.01 * c.grad
f.data+=0.01 * f.grad

e=a+b
d=e+c
L=d*f

print(L.data)


#creating a neuron
x1=Value(2.0,label='x1')
x2=Value(0.0,label='x2')

#weights
w1=Value(-3.0,label='w1')
w2=Value(1.0,label='w2')

#bias of the neuron
b=Value(6.8813735870195432,label='b')

x1w1=x1*w1;x1w1.label="x1w1"
x2w2=x2*w2;x2w2.label="x2w2"

x1w1x2w2=x1w1+x2w2;x1w1x2w2.label='x1w1+ x2w2'

n=x1w1x2w2+b;n.label="n"

#applying the activation function 
o=n.tanh()
o.label="o"
#for tanh we need to add a tanh() function to the Value class
draw_dot(o)

o.grad=1.0

#derivative of tanh
#d/dx tanh(x)=1-tanh(x)^2

# do/dn=1- o**2

1-o.data**2
#which is 0.5

n.grad=0.5

#n=x1w1x2w2 + b 
#grad of x1w1x2w2 and b will be equal to n.grad

x1w1x2w2.grad=0.5
b.grad=0.5

x1w1.grad=0.5
x2w2.grad=0.5

draw_dot(o)

x2.grad=w2.data * x2w2.grad
w2.grad=x2.data * x2w2.grad
draw_dot(o)

x1.grad=w1.data * x1w1.grad
w1.grad=x1.data * x1w1.grad

#coding the grad to automatically get calculated in the Value class itself

#creating a neuron
x1=Value(2.0,label='x1')
x2=Value(0.0,label='x2')

#weights
w1=Value(-3.0,label='w1')
w2=Value(1.0,label='w2')

#bias of the neuron
b=Value(6.8813735870195432,label='b')

x1w1=x1*w1;x1w1.label="x1w1"
x2w2=x2*w2;x2w2.label="x2w2"

x1w1x2w2=x1w1+x2w2;x1w1x2w2.label='x1w1+ x2w2'

n=x1w1x2w2+b;n.label="n"

#applying the activation function 
o=n.tanh()
o.label="o"
draw_dot(o)

#as for o out.grd=0
o.grad=1.0
o._backward()

n._backward()

#as it has no children it will do nothing and return none
b._backward()

x1w1x2w2._backward()
x1w1._backward()
x2w2._backward()
draw_dot(o)

#to automatically call backward on all the nodes ,we need to do for every node the baackward for all the nodes above it should be called


#this can be done using a topological sort

topo=[]
visited=set()

#add the node to list only after its children have been visited
def build_topo(v):
    if v not in visited:
        visited.add(v)
        for child in v._prev:
            build_topo(child)
        topo.append(v)

build_topo(o)
topo

#so we need to call backward in reverse topological order

#reset the gradients to 0 before the below code
o.grad=1
for node in reversed (topo):
    node._backward()

draw_dot(o)

#after adding backward to the Value class
o.backward()
draw_dot(o)

#the current backward fuction does not work for this case
a=Value(3.0,label=1)
b=a + a ;b.label="b"
b.backward()

#for this   when b calls _backward self and other both point to the same node so both self and other will set the gradient to 1 whereas the gradient should be 1+1

#so we need to accumulate the gradient so just replace = with +=



a=Value(3.0,label="a")
b=Value(2.0,label='b')
a/b
a-b
draw_dot(a-b)



x1=Value(2.0,label='x1')
x2=Value(0.0,label='x2')

#weights
w1=Value(-3.0,label='w1')
w2=Value(1.0,label='w2')

#bias of the neuron
b=Value(6.8813735870195432,label='b')

x1w1=x1*w1;x1w1.label="x1w1"
x2w2=x2*w2;x2w2.label="x2w2"

x1w1x2w2=x1w1+x2w2;x1w1x2w2.label='x1w1+ x2w2'

n=x1w1x2w2+b;n.label="n"
#implementing tanh using exponents and powers

#tanh=(e^2x-1)/(e^2x+1)
e=(2*n).exp()
o=(e - 1) / (e + 1)

o.label="o"
o.backward()
draw_dot(o)

import random
class Neuron:
    def __init__(self,nin):
        self.w=[Value(random.uniform(-1,1)) for _ in range (nin)]
        self.b=Value(random.uniform(-1,1))

    #forward pass
    def __call__(self,x):
        #takes two iterators and returns a new iterator that iterates over tuples
        
        act=sum((wi*xi for wi,xi in zip(self.w,x) ),self.b) 
        out=act.tanh()
        return out
    
    def parameters(self):
        return self.w+[self.b]


class Layer:
    def __init__(self,nin,nout):
        self.neurons=[Neuron(nin) for _ in range(nout)]
        
    def __call__(self,x):
        outs=[n(x)for n in self.neurons]
        return outs[0] if len(outs)==1 else outs
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
        
    

class MLP:
    #nouts is the list which contains the number of neurons in each layer
    #nins is the number of inputs 
    
    def __init__(self,nin,nouts):
        sz=[nin]+nouts
        self.layers=[Layer(sz[i],sz[i+1])for i in range (len(nouts))]
        
    def __call__(self,x):
        for layer in self.layers:
            x=layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


x=[2.0,3.0,-1.0]
n=MLP(3,[4,4,1])
n(x)
n.parameters() 
draw_dot(n(x))



#sample inputs
xs=[
    [2.0,3.0,-1.0],
    [3.0,-1.0,0.5],
    [0.5,1.0,1.0],
    [1.0,1.0,-1.0]
]

ys=[1.0,-1.0,-1.0,1.0] #desired outputs
y_preds=[n(x) for x in xs]
y_preds

#y ground truth(actual) and y out(predicted)
loss=sum([(yout-ygt)**2 for ygt,yout in zip(ys,y_preds)])

loss
loss.backward()
#now the neurons have a grad
n.layers[0].neurons[0].w[0]

draw_dot(loss)

#gradient descent 
for p in n.parameters():
    p.data += -0.01* p.grad 
    


#this below is the forward pass
y_preds=[n(x) for x in xs]
y_preds

#y ground truth(actual) and y out(predicted)
loss=sum([(yout-ygt)**2 for ygt,yout in zip(ys,y_preds)])

#backward pass
loss.backward()

draw_dot(loss)

#gradient descent 
for p in n.parameters():
    p.data += -0.01* p.grad 

loss




#making a training loop
n=MLP(3,[4,4,1])
n(x)
n.parameters() 



#sample inputs
xs=[
    [2.0,3.0,-1.0],
    [3.0,-1.0,0.5],
    [0.5,1.0,1.0],
    [1.0,1.0,-1.0]
]

ys=[1.0,-1.0,-1.0,1.0] #desired outputs
for k in range (10):
    #forward pass
    ypred=[n(x) for x in xs]
    loss=sum((yout-ygt)**2 for ygt ,yout in zip(ys,ypred))
    
    #backward pass
    for p in n.parameters():
        p.grad=0.0
    loss.backward()
    
    for p in n.parameters():
        p.data+=-0.05* p.grad
        
    print(k,loss.data)