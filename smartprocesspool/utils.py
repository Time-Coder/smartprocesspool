import sys
import warnings
import functools
import itertools
from types import ModuleType
from typing import Dict, Iterable

from .asizeof import asizeof


asizeof = functools.lru_cache(maxsize=None)(asizeof)

def _good_module_name(module_name:str, module_sizes:Dict[str, int])->bool:
    return (
        module_name and 
        module_name not in module_sizes and
        not module_name.startswith('builtins') and
        module_name in sys.modules
    )

def _get_module_sizes(module:ModuleType, module_sizes:Dict[str, int], get_size:bool)->Dict[str, int]:
    if get_size:
        module_sizes[module.__name__] = asizeof(module)
    else:
        module_sizes[module.__name__] = 0

    stack = [module]

    while stack:
        current_module = stack.pop()
        
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
                if get_size:
                    module_sizes[candidate_name] = asizeof(sub_module)
                else:
                    module_sizes[candidate_name] = 0

                stack.append(sub_module)
    
    return module_sizes

@functools.lru_cache(maxsize=None)
def get_module_sizes(module:ModuleType, get_size:bool=True)->Dict[str, int]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _get_module_sizes(module, {"sys": 0}, get_size)


def batched(iterable:Iterable, chunksize:int):
    iterator = iter(iterable)
    while True:
        batch = list(itertools.islice(iterator, chunksize))
        if not batch:
            break
        yield batch
