import os
import sys
import logging
import contextlib



### This module was part of a first-pass at seeing if Tensorflow output could be suppressed while the
### library was loaded and used by 'hex_finder.py', and much of this code was written by Gemini.
### This module, along with the approach in 'hex_finder.py' taken to tf/keras loading did reduce output, 
### but not completely. Some of this code may be doing nothing to achieve that goal!
# TODO: Rigorously pinpoint specifically what is working and why



@contextlib.contextmanager
def suppress_c_logs():
    """
    Redirects C-level standard output and standard error to /dev/null.
    This suppresses deep C++ logs (like XLA/Device mapping) that Python's
    sys.stdout/stderr capture cannot touch.
    """
    
    # Open the null device
    with open(os.devnull, "w") as devnull:
        # Save the original file descriptors
        try:
            # We use .fileno() to get the actual OS file descriptor
            original_stdout_fd = os.dup(sys.stdout.fileno())
            original_stderr_fd = os.dup(sys.stderr.fileno())
        except Exception:
            # If we are in an environment without real FDs (like some IDLE/Jupyter setups),
            # we simply yield and accept we can't suppress the C logs.
            yield
            return

        try:
            # Redirect stdout and stderr to null
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(devnull.fileno(), sys.stdout.fileno())
            os.dup2(devnull.fileno(), sys.stderr.fileno())
            
            yield
            
        finally:
            # Restore the original file descriptors
            os.dup2(original_stdout_fd, sys.stdout.fileno())
            os.dup2(original_stderr_fd, sys.stderr.fileno())
            
            # Close the duplicates
            os.close(original_stdout_fd)
            os.close(original_stderr_fd)


def import_tf_quietly():
    """
    Imports TensorFlow inside a C-level silence block.
    """
    
    # 1. Set environment variables (still helpful)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    # 2. Use the C-level suppressor context manager
    with suppress_c_logs():
        import tensorflow as tf
        # If you use keras directly, import it here too so it initializes quietly
        try:
            import keras
        except ImportError:
            pass # Keras might be part of tf.keras depending on version

        # 3. Silence Python-level logger
        logging.getLogger('tensorflow').setLevel(logging.FATAL)
        
    return tf


def quiet_function(func):
    """
    Decorator to run a function with C-level logs suppressed.
    Useful for 'lazy' TF initializations (like the first model build).
    """
    
    def wrapper(*args, **kwargs):
        with suppress_c_logs():
            return func(*args, **kwargs)
    return wrapper