pragma Singleton

import QtQuick 2.15

QtObject {
    property ControlSize titleBarHeight: ControlSize {
        main: 32
    }

    property ControlSize statusBarHeight: ControlSize {
        main: 26
    }


    property ControlSize controlWidth: ControlSize {
        main: 60
    }
    
    property ControlSize controlHeight: ControlSize {
        main: 28
    }

    property ControlSize buttonHeight : ControlSize {
        main: 28
    }
}