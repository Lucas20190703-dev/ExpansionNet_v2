import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

import KQuick.Core 1.0
import KQuick.Controls 1.0

Item {
	id: _root

	implicitWidth: 300
	implicitHeight: 300

	// menu button
	SideBar {
		id: _menuBar
		width: 42
		height: parent.height

		property real sidePosition: (_menuBar.width / _leftDrawer.width)
		opacity: 1 - Math.min(1.0, _leftDrawer.position / sidePosition)

		highlightIndex: _leftDrawer.currentIndex

		onClicked: (index) => {
			if (index === -1) { // home
				_leftDrawer.open();
			}
			else {
				_leftDrawer.currentIndex = index;
			}
		}
	}

	LeftDrawer {
		id: _leftDrawer
		y: 32
		width: 180
		height: parent.height - y
		currentIndex: 0
	}
	
	Connections {
		target: searchEngine.captionEngine

		function onDirCaptioningStarted(fileCount) {
			_statusBar.progressMessage = qsTr("Captioning");
			_statusBar.progress = 0;
		}

		function onDirCaptioningFinished() {
			_statusBar.message = "Captioning Finished."
		}

		function onProgressChanged(progress) {
			console.log("Progress changed:", progress)
			_statusBar.progress = progress;
		}
	}

	Connections {
		target: engineManager

		function onInitializingStarted() {
			_statusBar.progressMessage = qsTr("Captioning");
			_statusBar.message = qsTr("Initializing model...")
		}

		function onModelLoaded() {
			_statusBar.message = qsTr("Model loaded.")
		}
	}

	StatusBar {
		id: _statusBar
		anchors {
			bottom: parent.bottom
			left: _menuBar.right
			right: parent.right
		}
	}

	StackLayout {
		anchors {
			fill: parent
			leftMargin: _menuBar.width
			bottomMargin: _statusBar.height
		}

		currentIndex: _leftDrawer.currentIndex

		ImageSearchPage {

		}

		ImageCaptioningPage {

		}

		SettingsPage {

		}

		HelpPage {

		}

	}
}