import QtQuick 2.15
import QtQuick.Controls 2.15

import KQuick.Core 1.0

TextField {
	id: _control
	implicitWidth: 80
	implicitHeight: Size.controlHeight.main

	background: KRectangle {
		border {
			width: 1
			color: _control.activeFocus? Colors.control.border.highlight : Colors.control.border.primary
		}
	}

	color: Colors.control.foreground.primary

	opacity: enabled? 1.0 : 0.6

	selectedTextColor: Colors.control.foreground.primary

	selectByMouse: true
}