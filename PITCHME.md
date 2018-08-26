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


