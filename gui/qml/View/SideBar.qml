
import QtQuick 2.15

import KQuick.Controls 1.0
import KQuick.Core 1.0

Rectangle {
    id: _root

    signal clicked(int index)

    property int highlightIndex: 0

	width: 42	
    color: Colors.background.secondary

    Column {
        id: _menuBar
        width: parent.width

        component IconButtonDelegate : KIconButton {
            required property int index
            required property var iconSource

            width: parent.width
            height: parent.width
            highlighted: _root.highlightIndex == index

            icon {
                source: iconSource
                width: 16
                height: 16
                color: hovered || highlighted? Colors.accent.highlight : Colors.accent.primary
            }
            scale: hovered? 1.1 : 1.0

            Behavior on scale {
                NumberAnimation {
                    duration: 250
                    easing.type: Easing.InOutQuad
                }
            }

            onClicked: _root.clicked(index)
        }

        IconButtonDelegate {
            index: -1
            iconSource: "qrc:/icons/menu.png"
            icon.color: Colors.accent.primary
        }

        KSeparator {
            width: parent.width
        }

        Repeater {
            model: [
                "qrc:/icons/search.png",
                "qrc:/icons/image.png"
            ]

            IconButtonDelegate {
                required property var modelData
                iconSource: modelData
            }
        }

        KSeparator {
            width: parent.width
        }

        IconButtonDelegate {
            index: 2
            iconSource: "qrc:/icons/settings.png"
        }

        IconButtonDelegate {
            index: 3
            iconSource: "qrc:/icons/faq.png"
        }
    }
}