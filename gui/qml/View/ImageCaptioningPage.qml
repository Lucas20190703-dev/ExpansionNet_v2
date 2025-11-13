import QtQuick 2.15
import QtQuick.Controls 2.15

import KQuick.Core 1.0
import KQuick.Controls 1.0

BasePage {
	id: _root
	title: qsTr("Image Captioning")
	
	onVisibleChanged: {
		if (visible && d.imageSourceChanged) {
			_root.getCaption(_imagePreview.imageSource);
			d.imageSourceChanged = false;
		}
	}

	QtObject {
		id: d
		property bool imageSourceChanged: false
	}

	Connections {
		target: singleCaptioning

		function onReady() {
			_root.getCaption(_imagePreview.imageSource)
		}

		function onCaptionReady(captions) {
			console.log("Captions:", captions)
			_captionView.caption = captions.join('\n');
			_captionView.busy = false;
		}
	}

	SplitView {
		id: _hSplitView
		anchors.fill: parent
		
		handle: KSplitHandle {}
		clip: true

		FileListView {
			id: _sourceLIstView
			SplitView.fillHeight: true
			SplitView.preferredWidth: 200
			SplitView.minimumWidth: 160
		}

		Item {
			SplitView.fillHeight: true
			SplitView.fillWidth: true

			SplitView {
				id: _vSplitView
				anchors.fill: parent
				orientation: Qt.Vertical

				handle: KSplitHandle { }

				ImagePreview {
					id: _imagePreview
					SplitView.fillWidth: true
					SplitView.fillHeight: true
					imageSource: _sourceLIstView.currentImage

					onImageSourceChanged: {
						if (visible) {
							_root.getCaption(imageSource);
						}
						else {
							d.imageSourceChanged = true;
						}
					}
				}

				CaptionView {
					id: _captionView
					SplitView.fillWidth: true
					SplitView.preferredHeight: 150
					SplitView.minimumHeight: 80
					SplitView.maximumHeight: 200
				}
			}
		}
	}

	function getCaption(imageSource) {
		_captionView.busy = true;
		_captionView.caption = "";
		singleCaptioning.caption(imageSource);
	}
}