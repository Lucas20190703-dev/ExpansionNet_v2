pragma Singleton

import QtQuick 2.15

QtObject {
    id: _root

    property PaletteBasic background: PaletteBasic {
        primary: "#242424"
        secondary: "#202122"
        highlight: "#616161"
        disabled: "#80242424"
    }
    
    property PaletteBasic foreground: PaletteBasic {
        primary: "#ffffff"
        secondary: "#d0d0d0"
        highlight: "#ffffff"
        disabled: "#80ffffff"
    }
    
    property PaletteBasic border: PaletteBasic {
        primary: "#4d4d4d"
        secondary: "#804d4d4d"
        highlight: _root.accent.highlight
        disabled: "#404d4d4d"
    }

    property PaletteBasic accent: PaletteBasic {
        primary: "#f7d498"
        secondary: "#e2b260"
        highlight: "#f1bf67"
        disabled: "#9b7738"
    }

    property PaletteBasic drawer: PaletteBasic {
        primary: "#181818"
    }

    property PaletteControl control: PaletteControl {
        property PaletteBasic background: PaletteBasic {
            primary: "#202122"
            secondary: "#292828"
            highlight: "#363534" 
            disabled: "#80202122"
            focus: _root.accent.highlight
        }
        
        property PaletteBasic foreground: PaletteBasic {
            primary: "#e4e4e4"
            secondary: "#585858"
            highlight: "#ffffff"
            disabled: "#80ffffff"
            focus: _root.accent.highlight
        }
        
        property PaletteBasic border: PaletteBasic {
            primary: "#804d4d4d"
            secondary: "#804d4d4d"
            highlight: _root.accent.highlight
            disabled: "#404d4d4d"
        }
    }
}
