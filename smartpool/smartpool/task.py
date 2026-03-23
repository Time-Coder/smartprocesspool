from __future__ import annotations
import uuid
import sys
from concurrent.futures import Future
from typing import Tuple, Any, Dict, Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .worker import Worker


class Task:

    def __init__(self, func:Callable[..., Any], args:Tuple[Any, ...], kwargs:Dict[str, Any], need_cpu_cores:float, need_cpu_mem:float, need_gpu_cores:float, need_gpu_mem:float, calculate_module_deps:bool):
        self.id:str = str(uuid.uuid4())
        self.func:Callable[..., Any] = func
        self.args:Tuple[Any] = args
        self.kwargs:Dict[str, Any] = kwargs
        self.need_cpu_cores:int = need_cpu_cores
        self.need_cpu_mem:int = need_cpu_mem
        self.need_gpu_cores:int = need_gpu_cores
        self.need_gpu_mem:int = need_gpu_mem
        self.estimated_need_cpu_mem:float = 0.0
        self.modules_overlap_ratio:float = 0.0
        self.module_deps:Dict[str, int] = {}
        
        if calculate_module_deps:
            from .module_deps import module_deps
            self.module_deps:Dict[str, int] = module_deps(sys.modules[func.__module__])
        
        self.device:Optional[str] = None
        self.worker:Worker = None
        self.mem_before_enter:int = 0
        self.future = Future()

    def info(self):
        return self.id, self.device, self.func, self.args, self.kwargs

    @property
    def gpu_id(self)->int:
        if isinstance(self.device, str) and self.device.startswith("cuda:"):
            return int(self.device[len("cuda:"):])
        
        return -1

    def exec(self)->Tuple[bool, Any]:
        try:
            result = self.func(*self.args, **self.kwargs)
            success = True
        except BaseException as e:
            result = e
            success = False

        return success, result