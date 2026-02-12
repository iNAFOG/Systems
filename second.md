# What even is CUDA? and What the hell does a gpu looks like?

## A simple lookup into CUDA and GPU architecture

### GPU Architecture
Before Looking about CUDA and all those steps we should first look what a gpu looks like and its comparison to cpu with simple explanation.

A CPU  - and with cpu you must have heard about something called CORE - like "this is a quad core cpu or this is a octa core cpu".

#### What is a core?
A core in simple language is something like a small brain of CPU and a full cpu is made of up multiple core usually(in modern systems now at least).

A cpu core is designed in such a way that each can handle a complex task in itself.

#### Then what is a Thread?
You might have also heard about people saying 8 core 16 threads when taking about CPU so what is thread in all of this.

A thread is a most simple unit of program which is not dependent on others which so when some says 4 core 8 threads it usually means **each core of the cpu is designed to handle 2 threads in parallel.**

#### Where it all differs?

Each GPU have something called SMs(Streaming Multiprocessors) which is comparable to CPU cores (but not exactly). SMs are designed for parallelism.

*WARP* -  It is a collection of 32 threads bound together.
Each SM in gpu can track up to 64 warps(on NVIDIA Blackwell GPUs)  -  and each warp has 32 threads which means a single SM in gpu can 2048 concurrent threads.
This when compared to CPU where 2 thread running parallel in a core is common gpus can run up to 2048.

##### The Difference
The major difference is the kind of work they are designed to do, where the cpu cores are made to handle complex tasks a SMs is designed to doing simple task like adding two numbers. 

#### Some Key Terminologies
##### Thread, Thread blocks and Grids

At lowest level each thread executes your kernel code.
We can group threads (up to 1024 on modern gpus) together to form **thread block.**
**Thread Grid** is a collection of thread blocks.

#### Synchronization
First lets see - What is a Synchronization point?
It is a point where for example there are some thread in a thread block but they all need to perform some tasks and they need to sync their data and reach the same page so they can avoid any issues.
The point where they sync is the synchronization point.

In GPU it is done using _syncthreads().
But since we wait for synchronization and all threads to complete at the point we should avoid access synchronization.
The goal is to minimise those points.

#### Warp Divergence
Another thing to remember when we write out CUDA code is that - if some threads take "if" option and other take "else" it will induce something called warp divergence where every thread in a warp doesn't perform the same path execution and it add more total time in our execution time so its better to avoid it.

### Thank you
This was a pretty small one and i wrote this one quickly please correct if any error found in next part i will put some basic info on CUDA and some basic CUDA code explanation.

The topics discussed above are a starting point and you can start CUDA with the above topics knowledge and acquire more knowledge on complex topics as we go.