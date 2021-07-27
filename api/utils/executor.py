import asyncio
import multiprocessing
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
from functools import partial
from io import BytesIO
from typing import Callable, Optional


class _ProcessExecutor(object):

    _executor = None

    @classmethod
    def startup(cls):
        cls._executor = ThreadPoolExecutor()

    @classmethod
    def shutdown(cls):
        cls._executor.shutdown()

    @property
    def get(self):
        return self._executor

    async def run_task(self, func, *args, **kwargs):
        loop = asyncio.get_event_loop()
        func_task = partial(func, *args, **kwargs)
        result = await loop.run_in_executor(self._executor, func=func_task)
        return result

    async def run_in_shell(
        self,
        command: str,
        input_file: bytes,
        convert_output_func: Optional[Callable] = False,
    ):
        loop = asyncio.get_event_loop()
        proc = await asyncio.create_subprocess_shell(
            cmd=command,
            loop=loop,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate(input_file)
        if stdout:
            if convert_output_func:
                content = convert_output_func(stdout)
            else:
                content = BytesIO(stdout).getvalue()
            return content


executor = _ProcessExecutor()
