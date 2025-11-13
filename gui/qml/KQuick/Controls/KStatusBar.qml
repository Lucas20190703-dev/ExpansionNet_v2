import QtQuick 2.15
import QtQuick.Layouts 1.15

import KQuick.Core 1.0
import KQuick.Controls 1.0


Item {
    property alias filename: _filename.text
    property alias message: _message.text

    implicitWidth: 300
    implicitHeight: Size.statusBarHeight.main

    Rectangle {
        id: background
        anchors.fill: parent
        color: Colors.background.secondary
    }

    RowLayout {
        anchors.fill: parent
        anchors.leftMargin: 12
        anchors.rightMargin: 12

        KLabel {
            id: _filename
            Layout.preferredWidth: parent.width / 2
            Layout.alignment: Qt.AlignVCenter
        }

        KLabel {
            id: _message
            Layout.fillWidth: true
            Layout.alignment: Qt.AlignVCenter
            horizontalAlignment: KLabel.AlignRight
        }
    }
}