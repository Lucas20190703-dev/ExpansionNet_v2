import QtQuick 2.15
import KQuick.Core 1.0 

Item {
	id: _root

	property alias imageSource: _image.source

	readonly property size imageSize: Qt.size(_image.implicitWidth, _image.implicitHeight)

	Rectangle {
		anchors.fill: parent
		color: "transparent"

		border {
			width: 1
			color: Colors.control.border.primary
		}
	}

	Image {
		id: _image

		anchors {
			fill: parent
			margins: 1
		}

		fillMode: Image.PreserveAspectFit
	}
}