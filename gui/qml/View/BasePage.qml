import QtQuick 2.15
import QtQuick.Controls 2.15

import KQuick.Core 1.0
import KQuick.Controls 1.0

Page {
    id: _root
    padding: 9

    background: Rectangle {
		color: Colors.background.primary
	}

    header: Item {
        width: parent.width
        height: 26
        KLabel {
            anchors.centerIn: parent
            text: _root.title
            font {
                pointSize: 11
                weight: Font.DemiBold
            }
            color: Colors.foreground.primary
        }
    }
}