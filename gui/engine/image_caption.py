import os
from pathlib import Path

from PySide2.QtCore import QObject, Slot, Signal, Property, QThreadPool

import gui.engine.expansionnetv2_module as Module
import gui.engine.file_utils as Utils



from gui.engine.worker import Worker


class ImageCaptioning(QObject):
    
    ready = Signal()
    captionReady = Signal(list)
    
    def __init__(self, parent = None):
        super().__init__(parent)
        
        self.threadPool = QThreadPool()
        
    def get_caption(self, image_file):
        return Module.get_caption(image_file)
        
    @Slot(str)
    def caption(self, image_file):
        prefix = "file:///"
        if image_file.startswith(prefix):
            image_file = image_file[len(prefix):]
        if not image_file:
            self.captionReady.emit('')
            return
        
        worker = Worker(Module.get_caption, image_file)
        worker.signals.result.connect(lambda result: self.captionReady.emit(result[1]))
        self.threadPool.start(worker)

    

class DirectoryCaptioning(QObject):
    
    # signals
    rootDirChanged = Signal()
    captionReady = Signal(str, str)
    progressChanged = Signal(float)
    
    nameFilterChanged = Signal()
    
    dirCaptioningStarted = Signal(int)
    dirCaptioningFinished = Signal()
    
    
    def __init__(self, root = None, name_filters = ["*"], parent = None):
        super().__init__(parent)
        self._rootDir = root
        self._nameFilters = name_filters
        
        self.threadPool = QThreadPool()
            
        self._progress = 0
        self._processed = 0
        self._files = []
        self._database = None
        
        self.scan_files()
    
    def root_dir(self):
        return self._rootDir
    
    
    def set_root_dir(self, value : str):
        if self._rootDir == value:
            return
        
        self._rootDir = value
        self.rootDirChanged.emit()
        
        self.scan_files()


    def name_filters(self):
        return self._nameFilters
    
    
    def set_name_filters(self, value : list):
        self._nameFilters = value
        self.nameFilterChanged.emit()
        

    def progress_(self):
        return self._progress
    
    
    def set_database(self, database):
        self._database = database
    
    def scan_files(self):
        # gather files
        if self._rootDir:
            print(f"Scanning {self._nameFilters} files in {self._rootDir}")
            
            self._files = Utils.scan_files_recursively(self._rootDir, self._nameFilters)

            print(f"{len(self._files)} files are scanned.")
                    
    @Slot()
    def start(self):
        print ("Starting captioning...")
        self.dirCaptioningStarted.emit(len(self._files))
        
        self._progress = 0
        self._processed = 0
        
        for file in self._files:
            prefix = "file:///"
            if file.startswith(prefix):
                file = os.path.normpath(file[len(prefix):])
            
            if self._database and self._database.contains(filename = file):
                print(f"Caption exists. Skipping {file}")
                self.process_result(file, None, True)
                continue
            print ("Captioning thread start for:", file)
            worker = Worker(Module.get_caption, file)
            worker.signals.result.connect(lambda result: self.process_result(result[0], result[1]))
            self.threadPool.start(worker)
    
    @Slot(str, list)
    def process_result(self, file, result, exists=False):
        self._processed += 1
        self._progress = self._processed / len(self._files)
        self.progressChanged.emit(self._progress)
        
        print(f"Captioning ({round(self._progress * 100, ndigits=2)}%):", file)
        
        if not exists:        
            self.captionReady.emit(file, result)            
            self._database.insert(filename=file, captions=result)
        
        if len(self._files) == self._processed:
            self.dirCaptioningFinished.emit()
        
    @Slot()
    def close(self):
        self.threadPool.clear()
        self.threadPool.waitForDone()
        
    # properties
    rootDir = Property(str, root_dir, notify= rootDirChanged)
    progress = Property(str, progress_, notify=progressChanged)
    
