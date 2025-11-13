import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

import KQuick.Controls 1.0
import KQuick.Core 1.0

Pane {
    id: _root

    readonly property alias collapsed: _collapseButton.checked

    readonly property bool filtering: (_nameCheck.checked && _nameField.text) ||
                                    _dateCheck.checked ||
                                    (_captionCheck.checked && _captionCheck.text)

    readonly property bool nameFilterEnabled: _nameCheck.checked
    readonly property bool cpationFilterEnabled: _captionCheck.checked
    readonly property bool dateFilterEnabled: _dateCheck.checked

    readonly property string nameFilter: _nameCheck.checked? _nameField.text : ""
    readonly property string captionFilter: _captionCheck.checked? _captionArea.text : ""
    readonly property var startDateFilter: _dateCheck.checked? _dateStart.selectedDate : undefined
    readonly property var endDateFilter: _dateCheck.checked? _dateEnd.selectedDate : undefined
    readonly property real similarity: _similaritySlider.value


    background: Rectangle {
        color: Colors.background.primary
    }
    rightPadding: 16

    topPadding: collapsed? 10 : 16
    bottomPadding: 6

    implicitHeight: (collapsed? Math.max(32, _collapsedView.implicitHeight) : _grid.implicitHeight) + topPadding + bottomPadding

    Behavior on implicitHeight {
        NumberAnimation { duration: 300 }
    }
    GridLayout {
        id: _grid
        anchors.fill: parent
        columns: 4
        rowSpacing: 16
        columnSpacing: 6
        visible: !_root.collapsed

        KCheckBox {
            id: _nameCheck
            text: qsTr("Name")
        }

        KSearchField {
            id: _nameField
            Layout.preferredWidth: parent.width * 0.4
        }

        Item {
            Layout.fillWidth: true
            Layout.fillHeight: true
        }

        Row {
            Layout.fillHeight: true
            spacing: 12

            KCheckBox {
                id: _dateCheck
                anchors.verticalCenter: parent.verticalCenter
                text: qsTr("Date")
            }
            
            KDateField {
                id: _dateStart
                anchors.verticalCenter: parent.verticalCenter
                selectedDate: new Date(2000, 0, 1)
                onSelectedDateChanged: {
                    if (selectedDate > _dateEnd.selectedDate) {
                        _dateEnd.selectedDate = selectedDate;
                    }
                }
            }

            KLabel {
                anchors.verticalCenter: parent.verticalCenter
                text: "~"
            }

            KDateField {
                id: _dateEnd
                anchors.verticalCenter: parent.verticalCenter
                datePicker.x: _dateStart.width - datePicker.width
                onSelectedDateChanged: {
                    if (selectedDate < _dateStart.selectedDate) {
                        _dateStart.selectedDate = selectedDate;
                    }
                }
            }
        }
        
        
        KCheckBox {
            id: _captionCheck
            Layout.alignment: Qt.AlignTop 
            text: qsTr("Caption")
        }

        ScrollView {
            Layout.fillWidth: true
            Layout.preferredHeight: 64
            Layout.columnSpan: 3
            ScrollBar.vertical.policy: ScrollBar.AsNeeded
            
            KTextArea {
                id: _captionArea
            }
        }

        Row {
            Layout.column: 3
            Layout.row: 2
            Layout.alignment: Qt.AlignRight
            spacing: 2

            KLabel {
                anchors.verticalCenter: parent.verticalCenter
                text: qsTr("Similarity")
            }

            KSlider {
                id: _similaritySlider
                anchors.verticalCenter: parent.verticalCenter
                width: 120
                from: 0
                to: 1.0
                stepSize: 0.01
            }

            KLabel {
                anchors.verticalCenter: parent.verticalCenter
                text: _similaritySlider.value.toFixed(2)
                opacity: _captionCheck.checked? 1.0 : 0.6
            }
        }
    }

    Flow {
        id: _collapsedView
        anchors {
            left: parent.left
            leftMargin: 40
            right: parent.right
        }
        visible: _root.collapsed
        spacing: 6
        
        KRectangle {
            width: childrenRect.width + 16
            height: 32
            radius: 8
            visible: _nameCheck.checked
         
            Row {
                x: 10
                spacing: 10
                KLabel {
                    anchors.verticalCenter: parent.verticalCenter
                    text: _nameField.text                    
                }
                
                KIconButton {
                    anchors.verticalCenter: parent.verticalCenter
                    width: 24
                    height: 32
                    icon {
                        source: "qrc:/icons/close.png"
                        width: 10
                        height: 10
                        color: Colors.control.foreground.primary
                    }
                    onClicked: _nameCheck.checked = false
                }
            }
        }

        KRectangle {
            width: childrenRect.width + 16
            height: 32
            radius: 8
            visible: _dateCheck.checked
            
            Row {
                x: 10
                spacing: 10
                KLabel {
                    anchors.verticalCenter: parent.verticalCenter
                    text: _dateStart.text + "~" + _dateEnd.text                    
                }
                
                KIconButton {
                    anchors.verticalCenter: parent.verticalCenter
                    width: 24
                    height: 32
                    icon {
                        source: "qrc:/icons/close.png"
                        width: 10
                        height: 10
                        color: Colors.control.foreground.primary
                    }

                    onClicked: _dateCheck.checked = false
                }
            }
        }

        KRectangle {
            width: childrenRect.width + 16
            height: 32
            radius: 8
            visible: _captionCheck.checked
            
            Row {
                x: 10
                spacing: 10
                KLabel {
                    anchors.verticalCenter: parent.verticalCenter
                    width: Math.min(120, implicitWidth)
                    text: _captionArea.text
                    elide: KLabel.ElideRight
                }
                
                KIconButton {
                    anchors.verticalCenter: parent.verticalCenter
                    width: 24
                    height: 32
                    icon {
                        source: "qrc:/icons/close.png"
                        width: 10
                        height: 10
                        color: Colors.control.foreground.primary
                    }
                    onClicked: _captionCheck.checked = false
                }
            }
        }
    }

    KLabel {
        anchors {
            verticalCenter: parent.verticalCenter
            left: parent.left
            leftMargin: 40
        }

        text: qsTr("No filters")
        visible: !_nameCheck.checked && !_dateCheck.checked && !_captionCheck.checked && _root.collapsed
    }

    KIconButton {
        id: _collapseButton
        anchors {
            left: parent.left
            bottom: parent.bottom
        }
        width: 32
        height: 32
        icon {
            source: "qrc:/icons/down-double-arrow.png"
            width: 16
            height: 16
            color: Colors.control.foreground.highlight
        }
        checkable: true
        checked: true
        rotation: checked? 0 : 180
        Behavior on rotation {
            NumberAnimation { duration: 200 }
        }
    }
}