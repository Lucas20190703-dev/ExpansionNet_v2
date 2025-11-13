import QtQuick 2.15
import QtQuick.Layouts 1.15

import KQuick.Controls 1.0
import KQuick.Core 1.0

BasePage {
    title: qsTr("Home")
    padding: 0

    ColumnLayout {
        anchors.fill: parent
        spacing: 6

        Item {
            Layout.fillWidth: true
            Layout.preferredHeight: 42
            Row {
                x: 12
                height: parent.height
                
                KButton {
                    anchors.verticalCenter: parent.verticalCenter
                    text: qsTr("Root Directory")
                    icon {
                        source: "qrc:/icons/folder.svg"
                        width: 16
                        height: 16
                        color: Colors.foreground.highlight
                    }
                    flat: true
                    onClicked: _folderDialog.open()
                }
                
                KLabel {
                    anchors.verticalCenter: parent.verticalCenter
                    text: ":"
                }

                KLabel {
                    id: _directoryPath
                    anchors.verticalCenter: parent.verticalCenter
                    leftPadding: 10
                    color: Colors.foreground.primary
                }
            }
        }

        KSeparator {
            Layout.fillWidth: true
        }

        Item {
            Layout.fillWidth: true
            Layout.fillHeight: true
        }
    }
}