import QtQuick 2.15

ListView {
    id: _root

    property int currentYear: 2025
    signal selected(int year)

    model: 1000 // from 1900
    
    clip: true
    currentIndex: currentYear - 1900
    delegate: Item {
        width: _root.width
        height: 30
        KText {
            anchors.verticalCenter: parent.verticalCenter
            text: modelData + 1900
        }

        TapHandler {
            onTapped: {
                _root.selected(index + 1900);                
            }
        }
    }
}