import QtQuick 2.15
import QtQuick.Layouts 2.15
import KQuick.Core 1.0

Item {
    id: _root
    property alias model: _itemRepeater.model

    property int cellWidth: 160
    property int cellHeight: 90

    GridLayout {
        id: _gridView
        anchors.fill: parent
        anchors.margins: 9

        columns: Math.floor(width / _root.cellWidth)
        rowSpacing: 12

        Repeater {
            id: _itemRepeater
            delegate: KRectangle {
                Image {
                    anchors.fill: parent
                    sourceSize: Qt.size(width, height)
                    fillMode: Image.PreserveAspectFit
                    source: model.filename
                }
            }
        }
    }
}