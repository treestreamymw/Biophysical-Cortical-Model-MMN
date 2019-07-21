## adopted https://stackoverflow.com/questions/11312525/catch-ctrlc-sigint-and-exit-multiprocesses-gracefully-in-python/35134329#35134329
import multiprocessing as mp
import signal
import os


class PoolManager:
    def __init__(self, num_workers=None):
        default_num_workers = int((mp.cpu_count() / 2))
        self.num_workers = num_workers or default_num_workers
        self.original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)

    def worker(self, callable, map_args, timeout=60000000000000000000000000000000000000):

        pool = mp.Pool(processes=self.num_workers,  maxtasksperchild=1)

        signal.signal(signal.SIGINT, self.original_sigint_handler)

        try:
            n_jobs = len(map_args)

            print("Starting {} jobs using {} processes ".format(n_jobs,self.num_workers))
            res = pool.map_async(callable, map_args)

            # Without the timeout this blocking call ignores all signals.
            return res.get(timeout)

        except KeyboardInterrupt as e:

            print("Caught KeyboardInterrupt, terminating workers")
            raise e

        else:
            pool.close()




