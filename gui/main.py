import sys
import os
from pathlib import Path

from PySide2.QtGui import QGuiApplication, QSurfaceFormat, QIcon
from PySide2.QtQml import QQmlApplicationEngine
from PySide2.QtCore import Qt, QStandardPaths

sys.path.append(os.fspath(Path(__file__).parent.parent))

from engine.window_event_filter import WindowEventFilter
from gui.engine.manager import EngineManager

from resources import icons # rc file

def main():    
    # set graphic settings
    QGuiApplication.setAttribute(Qt.AA_UseOpenGLES)
    QGuiApplication.setAttribute(Qt.AA_EnableHighDpiScaling)

    format = QSurfaceFormat()
    format.setSwapBehavior(QSurfaceFormat.DoubleBuffer) # Double Buffering on
    format.setSwapInterval(0)                            # v-sync off
    QSurfaceFormat.setDefaultFormat(format)
    
    
    # create event filter for frameless window
    eventFilter = WindowEventFilter()
    
    # create application
    app = QGuiApplication(sys.argv)
    app.setApplicationName("Image Browser")
    app.setOrganizationName("Serhii Pyside2 Example")
    app.setOrganizationDomain("org.serhii.pyside.example")
    
    app.installNativeEventFilter(eventFilter)
    
    app.setWindowIcon(QIcon(":/icons/icon.ico"))
    
    font = app.font()
    font.setPointSize(9)
    app.setFont(font)
    
    # create qml engine
    engine = QQmlApplicationEngine()
    
    engine.addImportPath(str(Path(__file__).resolve().parent / "qml"))
    
    manager = EngineManager(engine)
    
    qml_file = Path(__file__).resolve().parent / "qml/Main.qml"
    
    def on_view_created(url):
        eventFilter.initWindow(engine)
        manager.initialize()
        
    engine.objectCreated.connect(on_view_created)
    
    engine.load(qml_file.as_uri())
    
    if not engine.rootObjects():

        sys.exit(-1)
    
    sys.exit(app.exec_())
    
if __name__ == "__main__":
    main()