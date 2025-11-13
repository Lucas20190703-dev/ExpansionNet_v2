import QtQuick 2.15
import QtQuick.Controls 2.15

import KQuick.Core 1.0
import KQuick.Controls 1.0

Item {
    id: _root
    readonly property alias text: _dateLabel.text

    property alias datePicker: _datePicker

    property var selectedDate: new Date()

    implicitWidth: 100
    implicitHeight: Size.controlHeight.main

    KRectangle {
        anchors.fill: parent
        radius: 2
    }

    KLabel {
        id: _dateLabel
        anchors {
            fill: parent
            leftMargin: 6
            rightMargin: 6
        }
        verticalAlignment: Qt.AlignVCenter
        text: dateString(_root.selectedDate)
        color: Colors.control.foreground.primary
    }
    
    TapHandler {
        onTapped: {
            if (_datePicker.visible) {
                _datePicker.close()
            }
            else {
                _datePicker.open();
            }
        }
    }

    Popup  {
        id: _datePicker

        x: parent.width / 2 - width / 2
        y: parent.height

        width: 360
        height: 240

        background: KRectangle {
            radius: 4
            opacity: 0.9
        }

        contentItem: KDatePicker {
            onClicked: (date) => {
                _root.selectedDate = date;
                _datePicker.close();
            }
        }
    }

    function dateString(date) {
        const month = String(date.getMonth() + 1).padStart(2, '0'); // Months are 0-based, so add 1 and pad with 0
        const day = String(date.getDate()).padStart(2, '0'); // Pad day with 0 if needed
        const year = date.getFullYear();
        const formattedDate = `${month}/${day}/${year}`;
        return formattedDate;
    }
}