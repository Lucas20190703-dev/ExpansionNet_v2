import QtQuick 2.15
import QtQuick.Controls 2.15

import KQuick.Core 1.0

Button {
    id: control

    property color textColor: control.down ? Colors.control.foreground.highlight : Colors.control.foreground.primary

    text: qsTr("Button")
    
    horizontalPadding: 6

    implicitWidth: contentItem.implicitWidth + leftPadding + rightPadding
    implicitHeight: Size.controlHeight.main

    contentItem: Row {
        spacing: 4
        KIcon {
            width: visible? height : 0
            height: parent.height
            icon {
                source: control.icon.source
                width: control.icon.width
                height: control.icon.height
                color: control.icon.color
            }
            visible: control.display === Button.IconOnly || control.display === Button.TextBesideIcon 
        }

        Text {
            anchors.verticalCenter: parent.verticalCenter
            text: control.text
            font: control.font
            opacity: enabled ? 1.0 : 0.3
            color: control.textColor
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            elide: Text.ElideRight
            visible: control.display === Button.TextOnly || control.display === Button.TextBesideIcon 
        }
    }

    background: KRectangle {
        implicitWidth: 100
        implicitHeight: 40
        opacity: enabled ? 1 : 0.3
        border.color: control.down? Colors.control.border.primary : Colors.control.border.secondary
        color: control.down ? Colors.control.background.highlight: Colors.control.background.primary
        visible: !control.flat
    }
}