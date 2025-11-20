###This is a numa-aware DDP script - more knowledge on it as we go down .


import os #python module to get,access files read, write and used to interact with operating system etc
import re #to search something in a string
import subprocess #module used to run shell commands
import psutil #view/management of system resources
import ctypes #library to use same shared c/c++ libraries
import torch #duh
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

_libnuma = ctypes.CDLL("libnuma.so")
if _libnuma.numa_available() < 0:
    raise RuntimeError("Numa not available")

#loads a library libnuma which as name suggests is a library related to numa
#checks for if it is available
#raises error if its not
#what is numa in all this ? - These are logical grouping of cpu and gpu - which are physically closer to each other
#example - lets say you have 4 gpus and you have 24 cpu you want to check which gpu is connected to which cpu with lowest latency and at the end ensure that if gpu 1 is connected to cpus 1-6 gpu1 should only exchange memory with those cpu
#why are we doing this - to reduce time - this process of doing or pinning those cpu - is called ? - yes you guess it right its called cpu pinning 

_libnuma.numa_run_on_node.argtypes = [ctypes.c_int]
_libnuma.numa_set_preferred.argtypes = [ctypes.c_int]

#we just set the prefered types

def parse_physical_cpu_list(phys_str: str):
    """Parse numactl --show physcpubind output like '0-3,8-11' into a list of ints."""
    cpus = []
    for part in phys_str.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            cpus.extend(range(start, end + 1))
        else:
            cpus.append(int(part))
    return cpus

#this is pretty self explanatory we just try to parse what will we part to this function
#what will be passed in the fxn here - output of the command numactl --show
#not whole output of numactl --show but physcpubind part of it.
#for better understanding run numactl --show on google colab terminal y'll get it.

