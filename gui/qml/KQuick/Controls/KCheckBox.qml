import QtQuick 2.15
import QtQuick.Controls 2.15

import KQuick.Core 1.0
import KQuick.Controls 1.0

CheckBox {
    id: control
    implicitHeight: Size.controlHeight.main
    
	indicator: Rectangle {
        implicitWidth: 18
        implicitHeight: 18
        x: control.leftPadding
        y: parent.height / 2 - height / 2
        radius: 3
        border.color: checked? Colors.control.border.highlight : Colors.control.border.primary
        color: Colors.control.background.primary

        KIcon {
            anchors.centerIn: parent
            icon {
                source: "qrc:/icons/check.png"
                width: 12
                height: 12
                color: Colors.control.border.highlight
            }
            visible: control.checked
        }
    }

    contentItem: Text {
        text: control.text
        font: control.font
        opacity: enabled ? 1.0 : 0.3
        color: checked? Colors.control.foreground.highlight : Colors.control.foreground.primary
        verticalAlignment: Text.AlignVCenter
        leftPadding: control.indicator.width + control.spacing
    }

}

