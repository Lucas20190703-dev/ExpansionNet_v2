import QtQuick 2.15
import QtQuick.Controls 2.15

import KQuick.Core 1.0

ProgressBar {
    id: control
    value: 0.5
    padding: 2

    background: KRectangle {
        implicitWidth: 200
        implicitHeight: 6
        radius: 4
    }

    contentItem: Item {
        implicitWidth: 200
        implicitHeight: 4

        // Progress indicator for determinate state.
        Rectangle {
            width: control.visualPosition * parent.width
            height: parent.height
            radius: 2
            color: Colors.control.foreground.primary
            visible: !control.indeterminate
        }

        // Scrolling animation for indeterminate state.
        Item {
            anchors.fill: parent
            visible: control.indeterminate
            clip: true

            Row {
                spacing: 20

                Repeater {
                    model: control.width / 40 + 1

                    Rectangle {
                        color: Colors.control.foreground.primary
                        width: 20
                        height: control.height
                    }
                }
                XAnimator on x {
                    from: 0
                    to: -40
                    loops: Animation.Infinite
                    running: control.indeterminate
                }
            }
        }
    }
}