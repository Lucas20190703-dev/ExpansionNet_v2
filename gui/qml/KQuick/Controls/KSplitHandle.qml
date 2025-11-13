import QtQuick 2.15

Rectangle {
	id: _handle
	property var orientation: parent.orientation

	implicitWidth: 6
	implicitHeight: 6
	color: "transparent"

	containmentMask: Item {
		x: (_handle.width - width) / 2
		y: (_handle.height - height) / 2
		width: _handle.orientation == Qt.Horizontal? 20 : _handle.width
		height: _handle.orientation == Qt.Horizontal? _handle.height : 20
	}
}