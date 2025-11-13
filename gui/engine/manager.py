
from PySide2.QtCore import *
from gui.engine.search_engine import SearchEngine
from gui.engine.image_caption import ImageCaptioning
from gui.engine.expansionnetv2_module import init_expansionnetv2_model
from gui.engine.worker import Worker

class EngineManager(QObject):
    initializingStarted = Signal()
    modelLoaded = Signal()
    
    def __init__(self, qmlEngine):
        super().__init__()
        
        self._searchEngine = SearchEngine()
        self._captionEngine = ImageCaptioning()
        
        qmlEngine.rootContext().setContextProperty("engineManager", self)
        qmlEngine.rootContext().setContextProperty("singleCaptioning", self._captionEngine)
        qmlEngine.rootContext().setContextProperty("searchEngine", self._searchEngine)
        
        pictureLocation = QStandardPaths.standardLocations(QStandardPaths.PicturesLocation)[0]
        qmlEngine.rootContext().setContextProperty("pictureLocation", pictureLocation)
        
        
    @Slot()
    def initialize(self):
        self.initializingStarted.emit()
        worker = Worker(init_expansionnetv2_model)
        worker.signals.finished.connect(self.on_model_loaded)
        QThreadPool.globalInstance().start(worker)        
    
    @Slot()
    def close(self):
        self._searchEngine.close()
        
    @Slot()
    def on_model_loaded(self):
        print("Model initialized.")
        self.modelLoaded.emit()
        self._searchEngine.start()