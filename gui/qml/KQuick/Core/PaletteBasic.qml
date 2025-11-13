import QtQuick 2.15

QtObject {
    property color primary
    property color secondary : primary
    property color highlight : primary
    property color disabled : primary
    property color focus : highlight
}