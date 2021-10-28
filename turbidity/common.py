import sys
import time
import functools
import logging

assert sys.version_info >= (3, 6)

# import tqdm
# class TqdmLoggingHandler(logging.Handler):
#     def __init__(self, level=logging.NOTSET):
#         super().__init__(level)

#     def emit(self, record):
#         try:
#             msg = self.format(record)
#             tqdm.tqdm.write(msg)
#             self.flush()
#         except (KeyboardInterrupt, SystemExit):
#             raise
#         except:
#             self.handleError(record)  

try:
    import coloredlogs, verboselogs

    logger = verboselogs.VerboseLogger('iwc')
    #level = 'SPAM'
    level = 'DEBUG'
    # logger.setLevel(level)
    coloredlogs.install(fmt='%(asctime)s %(message)s', level=level, logger=logger)

except ModuleNotFoundError:
    
    logging.basicConfig(format='%(asctime)s %(message)s')
    logger = logging.getLogger('iwc')
    logger.warning("Modules coloredlogs and/or verboselogs not found, using original logging module.")
    logger.setLevel(logging.DEBUG)

def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        logger.verbose(f"Finished {func.__name__!r} in {run_time:.4f} sec")
        return value
    return wrapper_timer