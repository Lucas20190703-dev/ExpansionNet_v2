import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

import KQuick.Core 1.0
import KQuick.Controls 1.0

Item {
    id: _root
    property string rootDir

    signal toggleSection(string section)

    onRootDirChanged: {
        searchEngine.rootFolder = rootDir
    }

    ListView {
        id: folderList
        anchors.fill: parent

        model: searchEngine.fileModel
        clip: true
        
        section.property: "directory"
        section.delegate: Rectangle {
            width: folderList.width
            height: 28
            color: Colors.background.highlight
            opacity: 0.6
            
            KLabel {
                anchors {
                    verticalCenter: parent.verticalCenter
                    left: parent.left
                    leftMargin: 16
                }
                opacity: 1.0
                text: section
            }

            KIconButton {
                anchors {
                    right: parent.right
                    rightMargin: 16
                }
                width: parent.height
                height: parent.height
                checkable: true
                icon {
                    source: "qrc:/icons/right-arrow.png"
                    width: 12
                    height: 12
                    color: Colors.control.foreground.primary
                }
                rotation: checked? 180 : 90
                Behavior on rotation {
                    NumberAnimation {}
                }

                onClicked: {
                    searchEngine.fileModel.toggleCollapse(section)
                }
            }
        }

        onContentYChanged: {
            for (let i = 0; i < folderList.count; i++) {
                const item = folderList.itemAtIndex(i);
                
                if (!item || !item.loadMoreTrigger || !item.visible)
                    continue;

                const triggerItem = item.loadMoreTrigger;
                const triggerY = triggerItem.mapToItem(folderList.contentItem, 0, 0).y;
                const buffer = 240;

                if (triggerY < folderList.contentY + folderList.height + buffer) {
                    const model = item.images;
                    if (model.canLoadMore)
                        model.loadNextBatch();
                }
            }
        }
        delegate: Column {
            id: _groupDelegate
            width: folderList.width

            property alias loadMoreTrigger : loadMoreTrigger
            required property var images
            required property int index
            required property bool collapsed

            Item {
                id: imageContainer
                width: parent.width
                height: _groupDelegate.collapsed? 1 : grid.implicitHeight + 12
                
                clip: true

                Behavior on height {
                    NumberAnimation{ duration: 500 }
                }
                
                Grid {
                    id: grid
                    property int cellWidth: 160 + 12
                    property int cellHeight: 90

                    anchors {
                        fill: parent
                        leftMargin: 16
                        rightMargin: 16
                        topMargin: 6
                        bottomMargin: 6
                    }

                    opacity: imageContainer.height / (implicitHeight + 12)

                    columns: Math.max(1, Math.floor(width / cellWidth))
                    columnSpacing: (columns < 2)? 10 : (width - columns * grid.cellWidth) / (columns - 1)
                    rowSpacing: 10

                    Repeater {
                        model: _groupDelegate.images

                        delegate: Column {
                            width: grid.cellWidth
                            spacing: 4

                            KRectangle {
                                anchors.horizontalCenter: parent.horizontalCenter
                                width: grid.cellWidth - 12
                                height: grid.cellHeight
                                color: "transparent"
                                
                                Image {
                                    anchors.fill: parent
                                    anchors.margins: 1
                                    fillMode: Image.PreserveAspectFit
                                    sourceSize: Qt.size(width, height)
                                    source: filePath
                                }

                                TapHandler {
                                    onDoubleTapped: {
                                        _previewPopup.imageSource = filePath
                                        _previewPopup.open()
                                    }
                                }
                            }

                            KText { 
                                anchors.horizontalCenter: parent.horizontalCenter
                                width: (parent.width - 12) * 0.8
                                text: fileName
                                font.pointSize: 8 
                                elide: Text.ElideRight
                            }
                        }
                    }
                }
            }
            Item {
                id: loadMoreTrigger
                width: 1
                height: 1
            }
        }

        ScrollBar.vertical: ScrollBar {
            
        }
    }
}