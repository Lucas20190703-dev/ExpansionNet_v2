
from PySide2.QtCore import *
from gui.engine.image_caption import DirectoryCaptioning
from gui.engine.image_list_model import FolderImageModel
from gui.engine.datafile import DataFile
from gui.engine.worker import Worker

from pathlib import Path

name_filters = ["jpg", "jpeg", "png", "bmp"]

class SearchEngine(QObject):
    
    rootFolderChanged = Signal()
    captionEngineChanged = Signal()
    fileModelChanged = Signal()
    
    
    
    def __init__(self, root_folder=None, parent=None):
        super().__init__(parent)
        
        self._captionEngine = DirectoryCaptioning(root=root_folder, name_filters=name_filters)
        self.captionEngineChanged.emit()
        
        self._fileModel = FolderImageModel()
        self.fileModelChanged.emit()
        
        self._rootFolder = root_folder
        self._database = None
        
        if root_folder:
            self.init(root_folder)
        
        
    def get_root_folder(self):
        return self._rootFolder
    
    def set_root_folder(self, val):
        if val.startswith("file:///"):
            val = val[8:]
            
        if self._rootFolder == val:
            return
        self._rootFolder = val
        self.rootFolderChanged.emit()
        
        if self._rootFolder:
            self.init(self._rootFolder)
        
    
    def get_caption_engine(self):
        return self._captionEngine

    def set_caption_engine(self, engine):
        if self._captionEngine == engine:
            return    
        self._captionEngine = engine
        self.captionEngineChanged.emit()
        
    def get_file_model(self):
        return self._fileModel
    
    def set_file_model(self, model):
        if self._fileModel == model:
            return
        
        self._fileModel = model
        self.fileModelChanged.emit()
        
    @Slot()
    def close(self):
        self._captionEngine.close()
        self._database.close()
    
    @Slot()
    def start(self):
        self.init_database(self._rootFolder)
        self._captionEngine.start()
        
    def init(self, root):
        self._rootFolder = root
        self._fileModel.loadFromPath(self._rootFolder)
        self._captionEngine.set_root_dir(self._rootFolder)
        
    
    def init_database(self, root):
        db_file = Path(root) / Path(root).name
        self._database = DataFile(db_file.as_posix())
        self._captionEngine.set_database(self._database)
        self._fileModel.setDatabase(self._database)
        
        
    @Slot(str, list)
    def add_to_database(self, file, captions):
        self._database.insert(filename=file, captions=captions)
        
        
    rootFolder = Property(str, fget=get_root_folder, fset=set_root_folder, notify=rootFolderChanged)
    fileModel = Property(QObject, fget=get_file_model, fset=set_file_model, notify=fileModelChanged)
    captionEngine = Property(QObject, fget=get_caption_engine, fset=set_caption_engine, notify=captionEngineChanged)
    
