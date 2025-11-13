import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import Qt.labs.platform 1.0 as L
import Qt.labs.settings 1.0

import KQuick.Controls 1.0
import KQuick.Core 1.0

BasePage {
    id: _root

    title: qsTr("Search")
    
    padding: 0

    ColumnLayout {
        anchors.fill: parent
        spacing: 0
        
        Item {
            Layout.fillWidth: true
            Layout.preferredHeight: 30
            Row {
                x: 12
                height: parent.height
                
                KButton {
                    anchors.verticalCenter: parent.verticalCenter
                    text: qsTr("Root Directory")
                    textColor: hovered? Colors.accent.highlight : Colors.accent.primary
                    icon {
                        source: "qrc:/icons/folder.svg"
                        width: 16
                        height: 16
                        color: hovered? Colors.accent.highlight : Colors.accent.primary
                    }
                    flat: true
                    onClicked: _folderDialog.open()
                }
                
                KLabel {
                    anchors.verticalCenter: parent.verticalCenter
                    text: ":"
                    color: Colors.accent.primary
                }

                KLabel {
                    id: _directoryPath
                    anchors.verticalCenter: parent.verticalCenter
                    leftPadding: 10
                    color: Colors.accent.primary
                }
            }
        }

        KSeparator {
            Layout.fillWidth: true
        }

        SearchPanel {
            id: _searchPanel
            Layout.fillWidth: true
            onNameFilterChanged: {
                searchEngine.fileModel.setSearchText(nameFilter);
            }

            onCaptionFilterChanged: {
                searchEngine.fileModel.setSearchCaption(captionFilter);
            }

            onSimilarityChanged: {
                searchEngine.fileModel.setSearchCaptionThreshold(similarity);
            }

            onStartDateFilterChanged: {
                searchEngine.fileModel.setSearchDate(startDateFilter, endDateFilter);
            }

            onEndDateFilterChanged: {
                searchEngine.fileModel.setSearchDate(startDateFilter, endDateFilter);
            }
        }
        
        KSeparator {
            Layout.fillWidth: true
        }

        ImageSearchContentView {
            id: _imageGrid
            Layout.fillWidth: true
            Layout.fillHeight: true
            rootDir: _directoryPath.text
        }
    }

    L.FolderDialog {
		id: _folderDialog
        folder: _settings.rootDir || "file:///" + pictureLocation
		onAccepted: {
			let dir = _folderDialog.folder.toString();
			if (dir.startsWith("file:///")) {
				dir = dir.substring(8);
			}
			_directoryPath.text = dir;
		}
	}

    Settings {
        id: _settings
        property alias rootDir: _directoryPath.text
    }

    Popup {
        id: _previewPopup
        property var imageSource

        anchors.centerIn: parent
        
        width: parent.width * 0.8
        height: parent.height * 0.8

        focus: true
        modal: true

        background: Rectangle {
            color: "transparent"
        }
        Overlay.modal: Rectangle {
            color: "#80000000"
        }

        contentItem: ColumnLayout {
            Image {
                id: _image
                Layout.fillWidth: true
                Layout.fillHeight: true
                fillMode: Image.PreserveAspectFit
                sourceSize: Qt.size(width, height)
                source: _previewPopup.imageSource
            }

            KLabel {
                Layout.alignment: Qt.AlignHCenter
                text: _image.source.toString().substring(8)
                font.pointSize: 11
            }
        }
    }

    function openPreviewImage(source) {
        _previewPopup.imageSource = source;
        _previewPopup.open()
    }
}