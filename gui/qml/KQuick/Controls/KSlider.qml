import QtQuick 2.15
import QtQuick.Controls 2.15

import KQuick.Core 1.0

Slider {
    id: control
    value: 0.5

    background: Rectangle {
        x: control.leftPadding
        y: control.topPadding + control.availableHeight / 2 - height / 2
        implicitWidth: 200
        implicitHeight: 3
        width: control.availableWidth
        height: implicitHeight
        radius: 2
        color: Colors.control.foreground.secondary
        opacity: control.enabled? 1.0 : 0.6
        
        Rectangle {
            width: control.visualPosition * parent.width
            height: parent.height
            color: Colors.control.foreground.highlight
            radius: 2
        }
    }

    handle: KRectangle {
        x: control.leftPadding + control.visualPosition * (control.availableWidth - width)
        y: control.topPadding + control.availableHeight / 2 - height / 2
        implicitWidth: 16
        implicitHeight: 16
        radius: 8
        color: control.pressed ? Colors.control.foreground.highlight : Colors.control.foreground.primary
        opacity: 1.0
        KToolTip {
            id: tooltip

            x: control.vertical ? control.spacing + parent.width + padding * 2 : (15 - width)/2
            y: control.horizontal ? -control.spacing - parent.height - padding * 2 : (15 - height)/2

            padding: 4
            opacity: 0.8
            visible: control.pressed

            parent: control.handle

            text: control.value.toFixed((control.stepSize + '.').split('.')[1].length)
            
            font.pixelSize: parent.width * 0.6
        }
    }
}