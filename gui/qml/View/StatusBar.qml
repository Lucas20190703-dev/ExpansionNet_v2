import QtQuick 2.15
import QtQuick.Layouts 1.15

import KQuick.Core 1.0
import KQuick.Controls 1.0


Item {
    id: _root

    property alias message: _messageLabel.text

    property alias progressMessage: _progressMessage.text
    property alias progress: _progressBar.value

    onProgressChanged: {
        if (progress == 1) {
            _progressTimer.restart();
        }
    }

    onMessageChanged: {
        _messageTimer.restart();
    }

    implicitWidth: 300
    implicitHeight: Size.statusBarHeight.main

    Rectangle {
        id: background
        anchors.fill: parent
        color: Colors.background.secondary
    }

    Row {
        anchors {
            left: parent.left
            leftMargin: 12
        }
        height: parent.height
        KLabel {
            id: _messageLabel
            anchors.verticalCenter: parent.verticalCenter
            width: 200            
        }
    }

    Row {
        anchors {
            right: parent.right
            rightMargin: 12
        }
        height: parent.height
        spacing: 6

        KLabel {
            id: _progressMessage
            anchors.verticalCenter: parent.verticalCenter
        }

        KProgressBar {
            id: _progressBar
            anchors.verticalCenter: parent.verticalCenter
            width: 120
            value: 0
        }

        KLabel {
            anchors.verticalCenter: parent.verticalCenter
            text: `${(_root.progress * 100).toFixed(2)}%`
            visible: _progressBar.visible
        }
    }

    Timer {
        id: _progressTimer
        interval: 3000
        repeat: false
        running: false

        onTriggered: {
            progress = 0;
            progressMessage = "";
            _progressBar.visible = false;
        }
    }

    Timer {
        id: _messageTimer
        interval: 10000
        repeat: false
        running: false

        onTriggered: {
            message = "";
        }
    }
}