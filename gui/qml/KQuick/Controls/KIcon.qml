import QtQuick 2.15
import QtGraphicalEffects 1.15

import KQuick.Controls 1.0

Item {
    id: _root

    property bool tintEnabled: true
    
    property KIconType icon : KIconType {}
	
    Image {
        id: _icon
        width: _root.icon.width
        height: _root.icon.height
        anchors.centerIn: parent
        source: _root.icon.source
        sourceSize: Qt.size(width * 2, height * 2)
        antialiasing: true
        fillMode: Image.PreserveAspectFit
        mipmap: true
    }

    ColorOverlay {
        anchors.fill: _icon
        source: _icon
        color: _root.icon.color
        visible: _root.tintEnabled
        enabled: _root.tintEnabled
    }
}
