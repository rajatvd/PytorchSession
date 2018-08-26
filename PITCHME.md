## Why PyTorch?
_Rajat V D_

This session provides a broad overview of how PyTorch is different, and why it might be a good idea to switch to it.

Use the very well written tutorial [here](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) for a detailed walkthrough of pytorch from scratch.
---
## Basics

@ul

* Treat tensors just like numpy arrays
* Use backward to find gradients
* Dynamic graphs, so everything executes as you call it (more later)

@ulend
---
## Neural networks

Pytorch follows the principles of _Object Oriented Programming_

@ul
* Build `modules`
* Combine them to form a big network
* Use torch apis like `data` and `optim` to train them
@ulend

+++
## Modules
The basic unit of any network is a module.

@ul
* You have to define the forward function
* autograd automatically defines the backward for you
* Just instantiate the module, and _call_ it on inputs to get your outputs
    * No sessions, placeholders or feed\_dicts
@ulend

+++
```
class Power(nn.Module):
    def __init__(self, exponent):
        self.exponent = exponent

    def forward(self, input):
        return input**self.exponent
```

@[1](Subclass nn.Module)
@[2,3](The init function is called when you make an instance. Set up instance variables)
@[5,6](`forward` takes in input tensor, and should return the output tensor)

+++
```
cube = Power(3)

inp = torch.randn(100)
output = cube(inp)
```

@[1](Create an instance of the module, passing in variables for the init funciton)
@[3,4](The instance is a callable, so just call it on tensors to get outputs)
+++

@ul
* Calling modules also builds the computation graph, so you can backward through them
* Allows you to easily modularize your network and code
* Modules can also have `parameters` which can be trained.
@ulend

+++
## Sub modules

@ul
* If you define modules in other modules, pytorch automatically recognizes them and adds them to a module list.
* The parameters of the parent module also include those of the submodule.
* Allows for hierarchical structure of networks
* Example - can have a hierarchy of modules like
    - one layer of convs with a residual connnection
    - a set of such layers with the same channel inputs and outputs
    - cascading such blocks with different channels
@ulend
---
## What about data?
@ul
* `torch.utils.data`
* Create a `Dataset` class which needs to have a `__getitem__` function.
* Make `DataLoader` objects from `Dataset`s which are iterators through the dataset.
* Can use multiprocessing to load data
@ulend

---
## Training
```
optimizer = torch.optim.SGD(net.parameters())
for images, labels in image_data_loader:
    outputs = net(images)
    loss = loss_fn(outputs, labels)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

@[1](Make an optimizer linked with your net's parameters)
@[2](Simple for loop over a `DataLoader`)
@[3,4](Forward pass to find loss function. This step builds the graph)
@[6](Remember that `backward` accumulates gradients, so zero before calling it)
@[7](Find the gradients of all leaves in the built graph w.r.t loss. This also destroys the built graph unless you specify to retain it)
@[8](Perform one optimization step using the calculated gradients)
---
## Dynamic graphs

@ul

* Build the computation graph on the fly
* The graph is built every time you run the model/module
* Gradients are also calculated dynamically

@ulend

+++
Benefits of dynamic graphs

@ul
* More intuitive to code
* Much easier to debug
    * Just treat it like a normal python program
* Can build some models which are impossible with static graphs

@ulend
---
## More pythonic you say?
It's an easy to answer to give when someone asks why pytorch - it's _pythonic_.

But what does that really mean?

+++
Good python is:

* Simple code.
* Easy to read, and easy to understand.
* Does what you would expect, even if it's the first time you're seeing it.
* Easy to debug.

Let's see how these play out for pytorch
+++


