import QtQuick 2.15

import KQuick.Core 1.0

Rectangle {
    color: Colors.control.background.primary
    border {
        width: 1
        color: Colors.control.border.primary
    }
    radius: 2
    opacity: enabled ? 1 : 0.4
}