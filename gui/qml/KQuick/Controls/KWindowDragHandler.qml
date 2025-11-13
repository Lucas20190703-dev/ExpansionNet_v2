import QtQuick 2.15
import QtQuick.Controls 2.15

// Allows dragging the window when placed on an unused section of the UI.
DragHandler {

    required property ApplicationWindow dragWindow

    target: null
    onActiveChanged: {
        if (active) dragWindow.startSystemMove()
    }
}
