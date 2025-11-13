import QtQuick 2.15
import QtQuick.Controls 2.15

import KQuick.Controls 1.0
import KQuick.Core 1.0

ComboBox {
    id: control
    model: ["First", "Second", "Third"]
    height: Size.controlHeight.main

    leftPadding: 6
    rightPadding: 6
    
    delegate: ItemDelegate {
        id: delegate

        //required property var model
        required property int index
        required property var modelData

        width: control.width
        height: 32

        background: Rectangle {
            color: delegate.highlighted? Colors.control.background.highlight : "transparent"
        }

        contentItem: Text {
            text: model[control.textRole] || modelData
            color: Colors.control.foreground.primary
            font: control.font
            elide: Text.ElideRight
            verticalAlignment: Text.AlignVCenter
        }
        highlighted: control.highlightedIndex === index
    }

    indicator: Canvas {
        id: canvas
        x: control.width - width - control.rightPadding
        y: control.topPadding + (control.availableHeight - height) / 2
        width: 12
        height: 8
        contextType: "2d"

        Connections {
            target: control
            function onPressedChanged() { canvas.requestPaint(); }
        }

        onPaint: {
            context.reset();
            context.moveTo(0, 0);
            context.lineTo(width, 0);
            context.lineTo(width / 2, height);
            context.closePath();
            context.fillStyle = control.pressed ? Colors.control.foreground.highlight : Colors.control.foreground.primary;
            context.fill();
        }
    }

    contentItem: Text {
        leftPadding: 0
        rightPadding: control.indicator.width + control.spacing

        text: control.displayText
        font: control.font
        color: control.pressed ? Colors.control.foreground.highlight : Colors.control.foreground.primary
        verticalAlignment: Text.AlignVCenter
        elide: Text.ElideRight
    }

    background: Rectangle {
        implicitWidth: 150
        implicitHeight: Size.controlHeight.main
        border.color: control.pressed || control.activeFocus ? Colors.control.border.highlight : Colors.control.border.primary;
        border.width: control.visualFocus ? 2 : 1
        radius: 2
        color: Colors.control.background.primary
    }

    popup: Popup {
        y: control.height - 1
        width: control.width
        height: contentItem.implicitHeight //Math.min(contentItem.implicitHeight, control.Window.height - topMargin - bottomMargin)
        padding: 1

        contentItem: ListView {
            clip: true
            implicitHeight: contentHeight
            model: control.popup.visible ? control.delegateModel : null
            currentIndex: control.highlightedIndex

            ScrollIndicator.vertical: ScrollIndicator { }
        }

        background: Rectangle {
            border.color: Colors.control.border.primary 
            radius: 2
            color: Colors.control.background.primary
        }
    }
}