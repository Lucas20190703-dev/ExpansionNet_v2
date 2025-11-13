import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

import KQuick.Core 1.0 
import KQuick.Controls 1.0

Drawer {
    id: _root

    property int currentIndex: 0
    leftInset: 1
    bottomInset: 1
    background: Rectangle {
        color: Colors.drawer.primary
        opacity: 0.8
    }

    Item {
        y: 20
        width: parent.width
        height: parent.height - y

        component MenuDelegate : ItemDelegate {
            id: _delegate
            
            required property var image
            required property var name
            required property var index

            width: parent.width
            height: 32

            background: Rectangle {
                color: _delegate.hovered? Colors.control.background.highlight : "transparent"
            }
            contentItem: Row {

                KIcon {
                    anchors.verticalCenter: parent.verticalCenter
                    width: 32
                    height: 32
                    icon {
                        source: _delegate.image
                        width: 16
                        height: 16
                        color: _delegate.hovered? Colors.accent.highlight : Colors.accent.primary
                    }
                }

                KLabel {
                    anchors.verticalCenter: parent.verticalCenter
                    text: _delegate.name
                    visible: _root.position > 0
                    color: _delegate.hovered? Colors.accent.highlight : Colors.accent.primary
                }
            }

            onClicked: {
                _root.currentIndex = _delegate.index;
                _root.close();
            }
        }

        Column {
            width: parent.width
            spacing: 6
            
            opacity: _root.position
            
            MenuDelegate {
                index: 0
                image: "qrc:/icons/search.png"
                name: qsTr("Search")
            } 

            MenuDelegate {
                index: 1
                image: "qrc:/icons/image.png"
                name: qsTr("Image Caption")
            } 

            KSeparator {
                width: parent.width
            }
            
            MenuDelegate {
                index: 2
                image: "qrc:/icons/settings.png"
                name: qsTr("Settings")
            }

            MenuDelegate {
                index: 3
                image: "qrc:/icons/faq.png"
                name: qsTr("Help")
            }
        }

        MenuDelegate {
            anchors{
                bottom: parent.bottom
                bottomMargin: 12
            }
            image: "qrc:/icons/close.png"
            name: qsTr("Exit")
            index: -1
            background: Rectangle {
                color: "transparent"
            }
            onClicked: {
                mainWindow.close()
            }
        }
    }
}