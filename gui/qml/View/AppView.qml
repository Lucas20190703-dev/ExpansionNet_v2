import QtQuick 2.15
import QtQuick.Controls 2.15

import KQuick.Core 1.0
import KQuick.Controls 1.0

Rectangle {
    id: _root

    property alias titlebar: _titlebar
    
    color: Colors.background.primary

    KTitleBar {
        id: _titlebar
        width: parent.width
        height: 30

        window: mainWindow

        onMinimize: {
            mainWindow.showMinimized()
        }

        onMaximize: {
            if (maximizeButton.checked) {
                mainWindow.showNormal()
            }
            else {
                mainWindow.showMaximized()
            }
        }

        onClose: {
            mainWindow.close();
        }
    }

    ContentView {
        anchors.fill: parent
        anchors.topMargin: _titlebar.height
    }
}