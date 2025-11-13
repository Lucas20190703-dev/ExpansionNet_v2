import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtQuick.Window 2.15

import KQuick.Core 1.0
import KQuick.Controls 1.0

Rectangle {
	id: _root

	property ApplicationWindow window

    readonly property var minimizeButton: _minimizeButton
    readonly property var maximizeButton: _maximizeButton
    readonly property var closeButton: _closeButton

    signal minimize
    signal maximize
    signal close

	implicitWidth: 300
	implicitHeight: Size.titleBarHeight.main

    color: Colors.background.secondary

    KIcon {
        id: _icon
        width: parent.height
        height: parent.height
        icon {
            width: 16
            height: 16
            source: "qrc:/icons/icon.ico"
            color: "transparent"
        }
    }

	KLabel {
		anchors {
            verticalCenter: parent.verticalCenter
            left: _icon.right
        }
		text: Qt.application.displayName
		font.pointSize: 12
        font.weight: Font.DemiBold
        color: Colors.foreground.primary
        opacity: _root.window.active? 1.0 : 0.6
	}

	KWindowDragHandler {
		dragWindow: _root.window
	}

	RowLayout {
        id: windowActions
        anchors.right: parent.right

        height: parent.height

        spacing: 0

        property int iconSize: parent.height - 18

        component InteractionButton: Rectangle {
            id: interactionButton

            signal action()

            property alias hovered: hoverHandler.hovered

            property bool checked: false

            onCheckedChanged: console.log("Checked changed:", checked)
            
            Layout.fillHeight: true
            Layout.preferredWidth: height * 1.2

            color: hovered ? Colors.control.background.highlight : "transparent"
            
            HoverHandler {
                id: hoverHandler
            }
            TapHandler {
                id: tapHandler
                onTapped: {
                    interactionButton.action();
                    interactionButton.checked = !interactionButton.checked;
                }
            }
        }

        InteractionButton {
            id: _minimizeButton

            onAction: _root.minimize()
            Rectangle {
                anchors.centerIn: parent
                color: Colors.foreground.primary
                height: 2
                width: windowActions.iconSize
            }
        }

        InteractionButton {
            id: _maximizeButton

            onAction: _root.maximize()

            Component.onCompleted: checked = _root.window.visibility == Window.Maximized
            
            Rectangle {
                anchors.centerIn: parent
                width: height
                height: windowActions.iconSize
                border.color: Colors.foreground.primary
                border.width: 1
                color: "transparent"
                visible: !_maximizeButton.checked
            }

            Rectangle {
                x: 11
                y: 9
                width: height
                height: windowActions.iconSize - 1
                border {
                    width: 1
                    color: Colors.foreground.primary
                }
                color: _root.color
                visible: _maximizeButton.checked
            }

            Rectangle {
                x: 9
                y: 11
                width: height
                height: windowActions.iconSize - 1
                border {
                    width: 1
                    color: Colors.foreground.primary
                }
                color: _root.color
                visible: _maximizeButton.checked
            }

        }

        InteractionButton {
            id: _closeButton

            color: hovered ? "#ec4143" : "transparent"
            onAction: _root.close()
            Rectangle {
                anchors.centerIn: parent
                width: windowActions.iconSize
                height: 2

                rotation: 45
                antialiasing: true
                transformOrigin: Item.Center
                color: Colors.foreground.primary

                Rectangle {
                    anchors.centerIn: parent
                    width: parent.height
                    height: parent.width

                    antialiasing: true
                    color: parent.color
                }
            }
        }
    }
}