import threading

class custom_thread(threading.Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        threading.Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    
    def run(self):
        if self._target is not None:    #  and self.stopped()
            self._return = self._target(*self._args,
                                                **self._kwargs)
    
    def join(self, *args):
        threading.Thread.join(self, *args)
        return self._return