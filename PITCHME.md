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
    * You can use native python conditonals and loops!
* autograd automatically defines the backward for you
* Just instantiate the module, and _call_ it on inputs to get your outputs
    * No sessions, placeholders or feed\_dicts
@ulend

+++
```
class Power(nn.Module):
    def __init__(self, exponent):
        super().__init__()
        self.exponent = exponent

    def forward(self, input):
        return input**self.exponent
```

@[1](Subclass nn.Module)
@[2,3,4](The init function is called when you make an instance)
@[3](Call the init of the superclass `nn.Module`)
@[4](Create an instance variable)
@[6,7](`forward` takes in input tensor, and should return the output tensor)

+++
```
cube = Power(3)

inp = torch.randn(100)
output = cube(inp)
```

@[1](Create an instance of the module, passing in stuff for the init funciton)
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
* Replicate your model across the batch dimension over multiple gpus with literally one line:
`model = nn.DataParallel(model)`
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

_Declarative vs Imperative_

@ul

* Computation = run
* The computation graph is built every time you run the model/module
* Gradients are also calculated dynamically


@ulend

+++
## Benefits of dynamic graphs

@ul
* Linear flow of the program
    * More intuitive to code
    * Much easier to debug - ust treat it like a normal python program
* Can build some models which are impossible with static graphs

@ulend
+++
## Examples of using dynamic graphs
* RNNs with variable length inputs are inherently dynamic
* Can build crazy networks like:
    * Have a random number of linear layers between 1 and 4 for each pass through the network
    * Make them share weights
+++
## Some problems

@ul
* Can't perform static optimizations\*
* Can't compile because you don't know what ops are going to be done
@ulend

\* torch jit can solve this problem
---
## I'll just use keras
* Of course, keras is much easier to code with.
* But it doesn't offer the flexibility of pytorch or tensorflow
* If you don't want to write for loops to train - just use my [utils](https://github.com/rajatvd/PytorchUtils)
    * Attempts to abstract out only the training part, while letting you still get comfy with the gradients.
    * No restrictive `fit` method - meaning you have to write the backward and step calls yourself
* Don't forget, keras doesn't have dynamic graphs too (tf.eager works, but I like my dynamic graphs without the baggage of 10 other APIs)

+++
## The verdict
@ul
* Pytorch is great for writing experiments and testing out ideas.
* It is super easy and fast to get the idea in your head to the GPUs in the workstation
* Not yet ideal for deploying stuff to production, but 1.0 will mostly change that with the jit
* For deploying highly optimized models for industry, tensorflow is probably the way to go, for now.
* Pytorch wins for research and experimentation.
@ulend
---
## Sacred Visdom 

Store and see everything about every run of your experiment

@ul
* `sacred` makes it extremely easy to write reproducible experiments. It keeps track of:
    * All the config variables you defined
    * Everything about the machine you ran the experiement on
    * Any metrics your experiment generated
* `visdom` is a visualization tool like tensorboard
    * You can use this or something like tensorboardX
    * I wrote a small [package](https://github.com/rajatvd/VisdomObserver) to integrate this with `sacred`

* Allows you to focus on the important part of your workflow - coding the experiments, not wasting time on saving stuff and writing code for plotting
* You don't have to use pytorch, this stuff is purely for improving your workflow, so go ahead and use this with tensorflow _shudders_

---
## Thank you
Check my github `rajatvd` for the slides and the other packages I talked about.
High quality tutorials for a __wide__ range of topics including RL, NLP, etc can be found in the official pytorch website [here](https://pytorch.org/tutorials/index.html)
Go through them to get an idea of you would implement specific types of models and networks.


The end