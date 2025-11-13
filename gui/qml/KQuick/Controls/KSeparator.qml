import QtQuick 2.15

import KQuick.Core 1.0

Item {
    property int orientation: Qt.Horizontal

    readonly property bool isHorizontal: orientation == Qt.Horizontal

    property int horizontalPadding: isHorizontal? 6 : 0
    property int verticalPadding: isHorizontal? 0 : 6


    implicitWidth: isHorizontal? 30 : 1 + horizontalPadding * 2
    implicitHeight: isHorizontal? 1 + verticalPadding * 2 : 30

    Rectangle {
        anchors.centerIn: parent
        width: isHorizontal? parent.width - horizontalPadding * 2 : 1
        height: isHorizontal? 1 : parent.height - verticalPadding * 2
        color: Colors.control.border.secondary
        opacity: 0.8
    }
}