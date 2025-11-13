import QtQuick 2.15
import QtQuick.Controls 2.15

import KQuick.Core 1.0

TextArea {
    id: _control
    implicitWidth: 300
    
    background: KRectangle {
		border {
			width: 1
			color: _control.activeFocus? Colors.control.border.highlight : Colors.control.border.primary
		}
	}

    wrapMode: TextArea.Wrap

    color: Colors.foreground.primary

	opacity: enabled? 1.0 : 0.6

	selectedTextColor: Colors.foreground.focus
	
	selectByMouse: true
}