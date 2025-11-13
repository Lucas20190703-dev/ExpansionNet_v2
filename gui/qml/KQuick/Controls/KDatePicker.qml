import QtQuick 2.15
import QtQuick.Controls 2.15
import KQuick.Core 1.0

ListView {
    id: _root
    

 // public
    function set(date) { // new Date(2019, 10 - 1, 4)
        selectedDate = new Date(date)
        positionViewAtIndex((selectedDate.getFullYear()) * 12 + selectedDate.getMonth(), ListView.Center) // index from month year
    }

    signal clicked(date date);  // onClicked: print('onClicked', date.toDateString())

 // private
    property date selectedDate: new Date()

    
    width: 500;  height: 400 // default size
    snapMode:    ListView.SnapOneItem
    orientation: Qt.Horizontal
    clip:        true

    model: 3000 * 12 // index == months since January of the year 0

    delegate: Item {        
        property int year:      Math.floor(index / 12)
        property int month:     index % 12 // 0 January
        property int firstDay:  new Date(year, month, 1).getDay() // 0 Sunday to 6 Saturday

        width: _root.width;  height: _root.height

        Column {
            Item { // month year header
                width: _root.width
                height: _root.height - grid.height

                KIconButton {
                    width: parent.height
                    height: parent.height
                    icon {
                        source: "qrc:/icons/right-arrow.png"
                        width: 12
                        height: 12
                        color: Colors.control.foreground.focus
                    }
                    rotation: 180

                    onClicked: {
                        let month = selectedDate.getMonth()
                        if (month > 0) {
                            _root.set(new Date(selectedDate.getFullYear(), month - 1, selectedDate.getDate()))
                        }
                        else {
                            const year = selectedDate.getFullYear() - 1;
                            month = 11;
                            _root.set(new Date(year, month, selectedDate.getDate()))    
                        }
                    }
                }
                KIconButton {
                    anchors {
                        right: parent.right
                    }
                    width: parent.height
                    height: parent.height
                    icon {
                        source: "qrc:/icons/right-arrow.png"
                        width: 12
                        height: 12
                        color: Colors.control.foreground.focus
                    }
                    onClicked: {
                        let month = selectedDate.getMonth()
                        if (month < 11) {
                            _root.set(new Date(selectedDate.getFullYear(), month + 1, selectedDate.getDate()))
                        }
                        else {
                            const year = selectedDate.getFullYear() + 1;
                            month = 0;
                            _root.set(new Date(year, month, selectedDate.getDate()))    
                        }
                    }
                }

                Row {
                    anchors.centerIn: parent
                    height: parent.height
                    spacing: 6
                    KText { // month year
                        id: monthLabel
                        anchors.verticalCenter: parent.verticalCenter

                        property var monthModel: ['January', 'February', 'March', 'April', 'May', 'June',
                            'July', 'August', 'September', 'October', 'November', 'December']
                        text: monthModel[month]
                        font {pixelSize: 0.5 * grid.cellHeight}

                        TapHandler {
                            onTapped: {
                                monthPicker.open()
                            }
                        }

                        Popup {
                            id: monthPicker
                            width: 100
                            height: 240
                            padding: 10

                            background: KRectangle {                               
                            }
                            contentItem: KMonthPicker {
                                month: selectedDate.getMonth()
                                onSelected: (month) => {
                                    _root.set(new Date(selectedDate.getFullYear(), month, selectedDate.getDate()))
                                }
                            }
                        }
                    }

                    KText { // month year
                        anchors.verticalCenter: parent.verticalCenter
                        text: year
                        font {pixelSize: 0.5 * grid.cellHeight}

                        TapHandler {
                            onTapped: {
                                yearPicker.open()
                            }
                        }

                        Popup {
                            id: yearPicker
                            width: 100
                            height: 240
                            padding: 10

                            background: KRectangle {
                                radius: 4
                            }

                            contentItem: KYearPicker {                                
                                currentYear: selectedDate.getFullYear()
                                onSelected: (year) => {
                                    _root.set(new Date(year, selectedDate.getMonth(), selectedDate.getDate()))
                                }
                            }
                        }
                    }
                }
            }

            Grid { // 1 month calender
                id: grid

                width: _root.width;  height: 0.875 * _root.height
                property real cellWidth:  width  / columns;
                property real cellHeight: height / rows // width and height of each cell in the grid.

                columns: 7 // days
                rows:    7

                Repeater {
                    model: grid.columns * grid.rows // 49 cells per month

                    delegate: KRectangle { // index is 0 to 48
                        property int day:  index - 7 // 0 = top left below Sunday (-7 to 41)
                        property int date: day - firstDay + 1 // 1-31

                        width: grid.cellWidth;  height: grid.cellHeight
                        border.width: 0.3 * radius
                        border.color: new Date(year, month, date).toDateString() == selectedDate.toDateString()  &&  text.text  &&  day >= 0?
                                      Colors.control.border.highlight : 'transparent' // selected
                        radius: 0.02 * _root.height
                        opacity: !mouseArea.pressed? 1: 0.3  //  pressed state

                        KText {
                            id: text
                            anchors.centerIn: parent
                            font.pixelSize: 0.5 * parent.height
                            font.bold:      new Date(year, month, date).toDateString() == new Date().toDateString() // today
                            text: {
                                if(day < 0)                                               ['S', 'M', 'T', 'W', 'T', 'F', 'S'][index] // Su-Sa
                                else if(new Date(year, month, date).getMonth() == month)  date // 1-31
                                else                                                      ''
                            }
                        }

                        MouseArea {
                            id: mouseArea

                            anchors.fill: parent
                            enabled:    text.text  &&  day >= 0

                            onClicked: {
                                selectedDate = new Date(year, month, date)
                                _root.clicked(selectedDate)
                            }
                        }
                    }
                }
            }
        }
    }

    Component.onCompleted: set(new Date()) // today (otherwise Jan 0000)
}