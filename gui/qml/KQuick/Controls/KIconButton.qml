import QtQuick 2.15
import QtQuick.Controls 2.15

ToolButton {
	id: _control

	display: Button.IconOnly
	flat: true
	
	implicitWidth: 30
	implicitHeight: 30

	background: Rectangle {
		color: "transparent"
	}

	contentItem: KIcon {
		icon.source: _control.icon.source
		icon.width: _control.icon.width
		icon.height: _control.icon.height
		icon.color: _control.icon.color
	}
}