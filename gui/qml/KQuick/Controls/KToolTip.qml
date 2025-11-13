import QtQuick 2.15
import QtQuick.Controls 2.15

import KQuick.Core 1.0

ToolTip {
    id: control

    opacity: 0.8

    contentItem: Text {
        text: control.text
        font: control.font
        color: Colors.control.foreground.primary
    }

    background: KRectangle {
        opacity: 0.8
    }

    delay: 100
    timeout: 0

}