from PySide2.QtCore import QRunnable, QThreadPool, QObject, Signal, Slot
import time
import sys

class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data, just emit when the processing is done
    
    error
        tuple (exctype, value, traceback.format_exc() )
    
    result
        object data returned from processing, anything
    
    progress
        int indicating % progress 
    '''
    finished = Signal()
    error = Signal(tuple)
    result = Signal(object)
    progress = Signal(int)

class Worker(QRunnable):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.
    '''

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @Slot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''
        
        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            # Catch any exception our task threw
            
            self.signals.error.emit(tuple(sys.exc_info()))
        else:
            # Return the result of the processing
            
            self.signals.result.emit(result)  # Return the result
        finally:
            self.signals.finished.emit()  # Done