from __future__ import annotations
import threading
from multiprocessing.connection import _ConnectionBase
import types
from typing import TYPE_CHECKING, Union, TypeAlias, Optional, Callable, Any, Dict, Tuple

import psutil

from .task import Task, Result
from .utils import comm_get, comm_put

if TYPE_CHECKING:
    try:
        import torch
        ConnectionOrQueue:TypeAlias = Union[torch.multiprocessing.Queue, _ConnectionBase]
    except ImportError:
        ConnectionOrQueue:TypeAlias = _ConnectionBase


class Worker:

    def __init__(
        self, index:int, result_queue:ConnectionOrQueue[Result], mp_context:str,
        initializer:Optional[Callable[..., Any]],
        initargs:Tuple[Any, ...],
        initkwargs:Optional[Dict[str, Any]],
        torch:bool=False
    ):
        if torch:
            import torch.multiprocessing as mp
        else:
            import multiprocessing as mp

        self.is_torch:bool = torch
        self.index:int = index
        self.mp_context:str = mp_context
        self.result_queue:ConnectionOrQueue[Result] = result_queue
        if torch:
            self.task_queue:ConnectionOrQueue[Task] = mp.Queue()
            self.sub_task_queue:ConnectionOrQueue[Task] = self.task_queue
        else:
            conn1, conn2 = mp.Pipe()
            self.task_queue:ConnectionOrQueue[Task] = conn1
            self.sub_task_queue:ConnectionOrQueue[Task] = conn2

        conn1, conn2 = mp.Pipe()
        self.change_device_cmd_connection:Optional[_ConnectionBase] = conn1
        self.sub_change_device_cmd_connection:Optional[_ConnectionBase] = conn2
        self._is_working:bool = False
        self._is_rss_dirty:bool = True
        self._cached_rss:int = 0
        self.module_sizes:Dict[str, int] = {}
        self.n_finished_tasks:int = 0
        self.initializer:Optional[Callable[..., Any]] = initializer
        self.initargs:Tuple[Any, ...] = initargs
        self.initkwargs:Optional[Dict[str, Any]] = initkwargs

        self.start()

    @property
    def cached_rss(self)->int:
        if self._is_working:
            return self.rss
            
        if self._is_rss_dirty:
            self._cached_rss = self.rss
            self._is_rss_dirty = False

        return self._cached_rss

    @property
    def rss(self)->int:
        try:
            return self.process_info.memory_info().rss
        except:
            return 0

    def overlap_modules_size(self, task:Task)->int:
        result = 0
        if len(task.module_sizes) < len(self.module_sizes):
            less_module_sizes = task.module_sizes
            more_module_sizes = self.module_sizes
        else:
            less_module_sizes = self.module_sizes
            more_module_sizes = task.module_sizes

        for module_name, module_size in less_module_sizes.items():
            if module_name in more_module_sizes:
                result += module_size

        return result

    @property
    def is_working(self)->bool:
        return self._is_working

    @is_working.setter
    def is_working(self, is_working:bool)->None:
        if self._is_working != is_working:
            self._is_working = is_working
            self._is_rss_dirty = True

    def add_task(self, task:Task)->None:
        self.is_working = True
        self.module_sizes.update(task.module_sizes)
        comm_put(self.task_queue, task)

    def change_device(self, device:str)->None:
        self.change_device_cmd_connection.send(device)

    def stop(self)->None:
        comm_put(self.task_queue, None)
        self.change_device_cmd_connection.send(None)

        self.task_queue.close()
        self.change_device_cmd_connection.close()

    def start(self):
        if self.is_torch:
            import torch.multiprocessing as mp
            Process = mp.Process
        else:
            import multiprocessing as mp
            ctx = mp.get_context(self.mp_context)
            Process = ctx.Process

        self.process:mp.Process = Process(
            target=Worker.run,
            args=(self.sub_task_queue, self.result_queue, self.sub_change_device_cmd_connection),
            kwargs={"initializer": self.initializer, "initargs": self.initargs, "initkwargs": self.initkwargs},
            name=f"SmartProcessPool.worker:{self.index}",
            daemon=True
        )
        self.process.start()
        self.process_info = psutil.Process(self.process.pid)

    def restart(self)->None:
        comm_put(self.task_queue, None)
        self.change_device_cmd_connection.send(None)
        self.process.join()
        self.n_finished_tasks:int = 0
        self.start()
        
    def terminate(self)->None:
        self.process.terminate()

    def join(self)->None:
        self.process.join()

    current_func = None
    func_device_lock = None
    change_device_thread = None

    @staticmethod
    def _changing_device(cmd_connection:_ConnectionBase[Optional[str]]):
        while True:
            device = cmd_connection.recv()
            if device is None:
                cmd_connection.close()
                break

            if Worker.current_func is None or Worker.current_func._device != "cpu":
                continue

            with Worker.func_device_lock:
                Worker.current_func._device = device

    @staticmethod
    def run(
        task_queue:ConnectionOrQueue[Task], result_queue:ConnectionOrQueue[Result], change_device_cmd_connection:_ConnectionBase[Optional[str]],
        initializer:Optional[Callable[..., Any]], initargs:Tuple[Any, ...], initkwargs:Optional[Dict[str, Any]]    
    ):
        if initializer is not None:
            if initkwargs is None:
                initkwargs = {}

            initializer(*initargs, **initkwargs)
        
        Worker.func_device_lock = threading.Lock()
        Worker.change_device_thread = threading.Thread(target=Worker._changing_device, args=(change_device_cmd_connection,), daemon=True, name="changing_device")
        Worker.change_device_thread.start()

        def device(self):
            with Worker.func_device_lock:
                return self._device

        while True:
            task:Optional[Task] = comm_get(task_queue)
            if task is None:
                task_queue.close()
                result_queue.close()
                break

            try:
                with Worker.func_device_lock:
                    task.func._device = task.device
                    task.func.device = types.MethodType(device, task.func)
                    Worker.current_func = task.func
            except AttributeError:
                pass

            comm_put(result_queue, task.exec())