import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Window 2.15

import View 1.0
import KQuick.Core 1.0

ApplicationWindow {
    id: mainWindow
    
    property int resizeBorderWidth: 6
    property var maximized: visibility == Window.Maximized

    property alias contentView: view

    visible: true
    width: 1080
    height: 760
    flags: Qt.Window | Qt.FramelessWindowHint | Qt.WindowMinMaxButtonsHint

    color: "transparent"
    
    onClosing: {
        engineManager.close()
    }
    
    AppView {
        id: view
        anchors.fill: parent
        anchors.margins: 1
    }

    Rectangle {
        anchors.fill: parent
        border {
            width: 1
            color: Colors.border.secondary
        }
        color: "transparent"
    }
}