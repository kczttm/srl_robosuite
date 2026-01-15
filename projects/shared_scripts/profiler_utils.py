import cProfile
import pstats
import io
import time
import os
from functools import wraps
from typing import Optional, Literal

try:
    from pyinstrument import Profiler as PyInstrumentProfiler
    HAS_PYINSTRUMENT = True
except ImportError:
    HAS_PYINSTRUMENT = False

class Profiler:
    """
    A versatile profiler that supports both cProfile and pyinstrument.
    Can be used as a context manager or a decorator.
    
    Usage:
        # As context manager
        with Profiler(backend='cprofile', output_file='profile.stats'):
            do_something()
            
        # As decorator
        @Profiler(backend='pyinstrument', output_file='profile.html')
        def my_func():
            pass
    """
    
    def __init__(
        self, 
        backend: Literal['cprofile', 'pyinstrument'] = 'cprofile', 
        output_file: Optional[str] = None,
        print_stats: bool = True,
        sort_by: str = 'cumulative',
        lines_to_print: int = 20
    ):
        self.backend = backend.lower()
        self.output_file = output_file
        self.print_stats = print_stats
        self.sort_by = sort_by
        self.lines_to_print = lines_to_print
        self.profiler = None
        
        if self.backend == 'pyinstrument':
            if not HAS_PYINSTRUMENT:
                print("Warning: pyinstrument not installed. Falling back to cProfile.")
                self.backend = 'cprofile'
            else:
                self.profiler = PyInstrumentProfiler()
        
        if self.backend == 'cprofile':
            self.profiler = cProfile.Profile()

    def __enter__(self):
        if self.backend == 'pyinstrument':
            self.profiler.start()
        else:
            self.profiler.enable()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.backend == 'pyinstrument':
            self.profiler.stop()
            if self.print_stats:
                print(self.profiler.output_text(unicode=True, color=True))
            
            if self.output_file:
                output_path = self.output_file
                if not output_path.endswith('.html'):
                    output_path += '.html'
                with open(output_path, 'w') as f:
                    f.write(self.profiler.output_html())
                print(f"Profile saved to {output_path}")
                
        else: # cProfile
            self.profiler.disable()
            s = io.StringIO()
            ps = pstats.Stats(self.profiler, stream=s).sort_stats(self.sort_by)
            
            if self.print_stats:
                ps.print_stats(self.lines_to_print)
                print(s.getvalue())
            
            if self.output_file:
                output_path = self.output_file
                if not output_path.endswith('.prof'):
                    output_path += '.prof'
                ps.dump_stats(output_path)
                print(f"Profile saved to {output_path}")
                # Suggest visualization
                print(f"To visualize: snakeviz {output_path}")

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper

def profile_function(backend='cprofile', output_file=None):
    """Decorator helper"""
    return Profiler(backend=backend, output_file=output_file)
