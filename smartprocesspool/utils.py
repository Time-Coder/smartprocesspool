from __future__ import annotations
import sys
from .asizeof import asizeof
# from pympler.asizeof import asizeof
import warnings
import functools
from enum import IntEnum
import itertools
from multiprocessing.connection import _ConnectionBase
from collections import deque
from types import ModuleType
from typing import Dict, Iterable, Any


class DataSize(IntEnum):
    B = 1
    KB = 1024
    MB = 1024 ** 2
    GB = 1024 ** 3
    TB = 1024 ** 4
    PB = 1024 ** 5
    

asizeof = functools.lru_cache(maxsize=None)(asizeof)

def _good_module_name(module_name:str, module_sizes:Dict[str, int])->bool:
    return (
        module_name and 
        module_name not in module_sizes and
        not module_name.startswith('builtins') and
        module_name in sys.modules
    )

def _get_module_sizes(module:ModuleType, module_sizes:Dict[str, int])->Dict[str, int]:
    module_sizes[module.__name__] = asizeof(module)
    stack = deque([module])

    while stack:
        current_module = stack.popleft()
        
        for obj in current_module.__dict__.values():
            candidate_name = ""
            if isinstance(obj, ModuleType) and hasattr(obj, '__name__'):
                candidate_name = obj.__name__
            
            elif hasattr(obj, '__module__'):
                candidate_name = obj.__module__
            
            elif hasattr(obj, '__class__') and hasattr(obj.__class__, '__module__'):
                candidate_name = obj.__class__.__module__

            if _good_module_name(candidate_name, module_sizes):
                sub_module = sys.modules[candidate_name]
                module_sizes[candidate_name] = asizeof(sub_module)
                stack.append(sub_module)
    
    return module_sizes

@functools.lru_cache(maxsize=None)
def get_module_sizes(module:ModuleType)->Dict[str, int]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _get_module_sizes(module, {"sys": 0})


def batched(iterable:Iterable, chunksize:int):
    iterator = iter(iterable)
    while True:
        batch = list(itertools.islice(iterator, chunksize))
        if not batch:
            break
        yield batch

def comm_put(queue, item)->None:
    if isinstance(queue, _ConnectionBase):
        queue.send(item)
    else:
        queue.put(item)

def comm_get(queue)->Any:
    if isinstance(queue, _ConnectionBase):
        return queue.recv()
    else:
        return queue.get()
