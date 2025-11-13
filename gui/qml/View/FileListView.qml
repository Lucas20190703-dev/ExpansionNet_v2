import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

import Qt.labs.qmlmodels 1.0
import Qt.labs.folderlistmodel 2.15
import Qt.labs.platform 1.0 as L
import Qt.labs.settings 1.0

import KQuick.Core 1.0 
import KQuick.Controls 1.0


Item {
	id: _root
	property var rootDir: ""

	property string title: qsTr("Source")

	readonly property var currentImage: "file:///" + (_listView.currentItem? _listView.currentItem.filePath : "")

	Component.onDestruction: {
		_settings.currentIndex = _listView.currentIndex;
	}
	
	Settings {
		id: _settings
		property alias sourceLocation: _root.rootDir
		property int currentIndex: 0
	}

	Shortcut {
		sequence: "Down"
		onActivated: _listView.currentIndex = Math.min(_listView.currentIndex + 1, _folderModel.count - 1)
	}

	Shortcut {
		sequence: "Up"
		onActivated: _listView.currentIndex = Math.max(_listView.currentIndex - 1, 0)
	}

	Rectangle {
		anchors.fill: parent
		color: Colors.background.primary
		border {
			width: 1
			color: Colors.control.border.primary
		}
	}

	ColumnLayout {
		anchors {
			fill: parent
			margins: 1
		}
		spacing: 0

		Rectangle {
			Layout.fillWidth: true
			Layout.preferredHeight: 26
			color: Colors.background.secondary

			MouseArea {
				id: _folderButton
				width: _buttonContent.width
				height: parent.height
				hoverEnabled: true

				onClicked: {
					_folderDialog.open();
				}

				Row {
					id: _buttonContent
					height: parent.height
					KIcon {
						anchors.verticalCenter: parent.verticalCenter
						width: 24
						height: 24
						icon.source: "qrc:/icons/folder.svg"
						icon.width: 16
						icon.height: 16
						icon.color: _folderButton.containsMouse? Colors.accent.highlight : Colors.accent.primary
					}

					KText {
						anchors.verticalCenter: parent.verticalCenter
						text: _root.rootDir? _root.rootDir : "Select root folder"
						color: _folderButton.containsMouse? Colors.accent.highlight : Colors.accent.primary
					}
				}
			}
		}

		ListView {
			id: _listView
			Layout.fillWidth: true
			Layout.fillHeight: true

			clip: true

			model: _folderModel

			delegate: Item {
				id: _delegateItem
				required property var fileName
				required property var filePath
				required property int index

				width: _listView.width
				height: 32
				
				KText {
					anchors.verticalCenter: parent.verticalCenter
					width: parent.width
					text: _delegateItem.fileName
					elide: Text.ElideLeft
					leftPadding: 6
					rightPadding: 6
				}
				
				TapHandler {
					onTapped: {
						console.log(`Item ${_delegateItem.index + 1} selected.`)
						_listView.currentIndex = _delegateItem.index;
					}
				}
			}

			highlight: Rectangle {
				color: Colors.control.background.highlight
			}

			ScrollBar.vertical: ScrollBar {
				policy: ScrollBar.AsNeeded
				active: ScrollBar.AlwaysOn
			}
		}
	}

	FolderListModel {
		id: _folderModel
		rootFolder: folder
		showFiles: true
		showDirs: false

		folder: "file:///" + _root.rootDir
		nameFilters: ["*.jpg", "*.jpeg", "*.png"]

		onStatusChanged: {
			if (_folderModel.status == FolderListModel.Ready) {
				_listView.currentIndex = _settings.currentIndex;
			}
		}
	}

	L.FolderDialog {
		id: _folderDialog
		folder: _root.rootDir || pictureLocation
		onAccepted: {
			let dir = _folderDialog.folder.toString();
			if (dir.startsWith("file://")) {
				dir = dir.substring(8);
			}
			_root.rootDir = dir;
		}
	}

	function getFiles() {
		let files = [];
		for (let i = 0; i < _folderModel.count; i++) {
			const filePath = _folderModel.get(i, "filePath");
			files.push(filePath);
		}
		return files;
	}
}