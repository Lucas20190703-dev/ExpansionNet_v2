import QtQuick 2.15

ListView {
    id: _root

    property int month: 0
    signal selected(int index)

    model: ['January', 'February', 'March', 'April', 'May', 'June',
                            'July', 'August', 'September', 'October', 'November', 'December']

    currentIndex: month
    clip: true
    delegate: Item {
        width: _root.width
        height: 30
        KText {
            anchors.verticalCenter: parent.verticalCenter
            text: modelData
        }

        TapHandler {
            onTapped: {
                _root.selected(index);                
            }
        }
    }
}