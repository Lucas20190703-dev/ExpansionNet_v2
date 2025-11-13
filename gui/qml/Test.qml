import QtQuick 2.15
import Qt.labs.folderlistmodel 2.15

Item {
    width: 960
    height: 640

    ListView {
        anchors.fill: parent

        model: _folderModel
        delegate: Text {
            text: model.filePath
        }

    }

    FolderListModel {
        id: _folderModel
        showDirs: true
        showFiles: false

        folder: "file:///E:\\"
    }
}