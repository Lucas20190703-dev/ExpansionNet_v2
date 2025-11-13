import QtQuick 2.15
import QtQuick.Layouts 1.15

import KQuick.Controls 1.0

BasePage {
    title: qsTr("Settings")
    
    Item {
        anchors.fill: parent
        anchors.margins: 40

        GridLayout {
            columns: 2
            rowSpacing: 24
            columnSpacing: 8

            KLabel {
                text: qsTr("Theme")
            }

            KComboBox {
                model: ["Dark", "Light", "Blue"]
            }

            KLabel {
                text: qsTr("Language")
            }

            KComboBox {
                model: ["English"]
            }

            KLabel {
                text: qsTr("Engine")
            }

            KComboBox {
                model: ["ExpansionNet_V2"]
            }
        }
        
    }
}