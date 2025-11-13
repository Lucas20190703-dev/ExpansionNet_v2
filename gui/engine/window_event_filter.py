from PySide2.QtQml import QQmlApplicationEngine
from PySide2.QtCore import QObject, QAbstractNativeEventFilter, QEvent, Signal, Slot
from PySide2.QtQuick import QQuickWindow
from PySide2.QtGui import QScreen

from engine.c_structures import *

from ctypes import POINTER, byref, c_bool, c_int, pointer, sizeof, WinDLL, cast

from ctypes.wintypes import DWORD, LONG, LPCVOID, LPRECT, MSG

import ctypes

import win32gui
import win32con
import win32api

class WindowEventFilter(QObject, QAbstractNativeEventFilter):
    
    def __init__(self, parent = None, engine = None):
        QObject.__init__(self, parent)
        QAbstractNativeEventFilter.__init__(self)
        
        self.quick_window = None
        self.hwnd = None
        self.engine = engine
        self.resize_border_width = 6
        
        self.user32 = WinDLL("user32")
        self.dwmapi = WinDLL("dwmapi")
        
        self.DwmExtendFrameIntoClientArea = self.dwmapi.DwmExtendFrameIntoClientArea 
        
        if self.engine:
            self.initWindow(self.engine)
        
        
    def initWindow(self, engine : QQmlApplicationEngine) -> bool:
        self.quick_window = engine.rootObjects()[0]
        
        if not self.quick_window:
            return False
        
        self.hwnd = (self.quick_window.winId())
        self.resize_border_width = int(self.quick_window.property("resizeBorderWidth")) * self.quick_window.devicePixelRatio()
        
        self.quick_window.screenChanged.connect(self.onScreenChanged)
        
        # Set window shadows.
        margins = MARGINS(1, 1, 1, 1)
        self.DwmExtendFrameIntoClientArea(self.hwnd, margins)
        
        engine.installEventFilter(self)
        
        engine.rootContext().setContextProperty("windowEventFilter", self)

        return True
    
    @Slot(QScreen, result=None)
    def onScreenChanged(self, screen):
        self.user32.SetWindowPos(self.hwnd, None, 0, 0, 0, 0,
            win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_NOZORDER |
            win32con.SWP_NOOWNERZORDER | win32con.SWP_FRAMECHANGED | win32con.SWP_NOACTIVATE)
    
    def eventFilter(self, watched, event):
        # match event.type():
        #     case QEvent.Enter:
        #         pass
        #     case QEvent.Leave:
        #         pass
        #     case _:
        #         pass
            
        return QObject.eventFilter(self, watched, event)
    
    @classmethod
    def get_window_placement(self, hwnd) -> WINDOWPLACEMENT:
        wp = WINDOWPLACEMENT()
        wp.length = sizeof(WINDOWPLACEMENT)
        ctypes.windll.user32.GetWindowPlacement(hwnd, byref(wp))
        return wp
    
    @classmethod
    def get_window_rect(self, hwnd) -> RECT:
        rect = RECT()
        
        ctypes.windll.user32.GetWindowRect(hwnd, byref(rect))
        return rect
    
    @classmethod
    def get_x_lparam(self, lparam):
        return lparam & 0xFFFF  # Low-order word
    
    @classmethod
    def get_y_lparam(self, lparam):
        return (lparam >> 16) & 0xFFFF  # High-order word
        
    def nativeEventFilter(self, eventType, message):
        if eventType == "windows_generic_MSG":
            msg = MSG.from_address(message.__int__())
            
            if not msg.hWnd:
                return False
            
            match msg.message:
                case win32con.WM_NCCALCSIZE:
                    if msg.lParam:
                        wp = self.get_window_placement(self.hwnd)
                        if wp.showCmd == SW_MAXIMIZE:
                            sz = cast(msg.lParam, POINTER(NCCALCSIZE_PARAMS)).contents
                            # sz.rgrc[0].left += PADDING_SIZE
                            # sz.rgrc[0].top += PADDING_SIZE
                            # sz.rgrc[0].right -= PADDING_SIZE
                            # sz.rgrc[0].bottom -= PADDING_SIZE
                            
                            # Assuming m_quick_window is a reference to your UI framework
                            # maximize_button = QObject(self.quick_window.findChild(QObject, "maximizeButton"))
                            # maximize_button.setProperty("checked", True)
                        elif wp.showCmd == SW_NORMAL:
                            # maximize_button = QObject(self.quick_window.findChild(QObject, "maximizeButton"))
                            # maximize_button.setProperty("checked", False)
                            x = 0
                    return True
                case win32con.WM_NCHITTEST:
                    winrect = self.get_window_rect(msg.hWnd)
                    x = self.get_x_lparam(msg.lParam)
                    y = self.get_y_lparam(msg.lParam)
                    
                    left = winrect.left
                    top = winrect.top
                    right = winrect.right
                    bottom = winrect.bottom
                    # Hit testing logic
                    if (x >= left and x < left + self.resize_border_width and
                        y < bottom and y >= bottom - self.resize_border_width):
                        return True, win32con.HTBOTTOMLEFT
                    if (x < right and x >= right - self.resize_border_width and
                        y < bottom and y >= bottom - self.resize_border_width):
                        return True, win32con.HTBOTTOMRIGHT
                    if (x >= left and x < left + self.resize_border_width and
                        y >= top and y < top + self.resize_border_width):
                        return True, win32con.HTTOPLEFT
                    if (x < right and x >= right - self.resize_border_width and
                        y >= top and y < top + self.resize_border_width):
                        return True, win32con.HTTOPRIGHT
                    if x >= left and x < left + self.resize_border_width:
                        return True, win32con.HTLEFT
                    if x < right and x >= right - self.resize_border_width:
                        return True, win32con.HTRIGHT
                    if y < bottom and y >= bottom - self.resize_border_width:
                        return True, win32con.HTBOTTOM
                    if y >= top and y < top + self.resize_border_width:
                        return True, win32con.HTTOP
                    return False, win32con.HTTRANSPARENT                
                case win32con.WM_SYSCOMMAND:
                    if msg.wParam == win32con.VK_SPACE or (msg.wParam == win32con.SC_KEYMENU and msg.lParam == win32con.VK_SPACE):
                        menu = self.user32.GetSystemMenu(msg.hWnd, False)
                        if menu:
                            mii = MENUITEMINFO()
                            mii.cbSize = ctypes.sizeof(MENUITEMINFO)
                            mii.fMask = win32con.MIIM_STATE
                            mii.fType = 0
                            
                            mii.fState = win32con.MF_ENABLED
                            self.user32.SetMenuItemInfo(menu, win32con.SC_MINIMIZE, False, ctypes.byref(mii))
                            self.user32.SetMenuItemInfo(menu, win32con.SC_SIZE, False, ctypes.byref(mii))
                            self.user32.SetMenuItemInfo(menu, win32con.SC_MOVE, False, ctypes.byref(mii))
                            self.user32.SetMenuItemInfo(menu, win32con.SC_MAXIMIZE, False, ctypes.byref(mii))
                            self.user32.SetMenuItemInfo(menu, win32con.SC_MINIMIZE, False, ctypes.byref(mii))
                            
                            mii.fState = win32con.MF_GRAYED

                            wp = self.get_window_placement(self.hwnd)
                            
                            if wp.showCmd == win32con.SW_SHOWMAXIMIZED:
                                self.user32.SetMenuItemInfo(menu, win32con.SC_SIZE, False, ctypes.byref(mii))
                                self.user32.SetMenuItemInfo(menu, win32con.SC_MOVE, False, ctypes.byref(mii))
                                self.user32.SetMenuItemInfo(menu, win32con.SC_MAXIMIZE, False, ctypes.byref(mii))
                                self.user32.SetMenuDefaultItem(menu, win32con.SC_CLOSE, False)
                            elif wp.showCmd == win32con.SW_SHOWMINIMIZED:
                                self.user32.SetMenuItemInfo(menu, win32con.SC_MINIMIZE, False, ctypes.byref(mii))
                                self.user32.SetMenuDefaultItem(menu, win32con.SC_RESTORE, False)
                            elif wp.showCmd == win32con.SW_SHOWNORMAL:
                                self.user32.SetMenuItemInfo(menu, win32con.SC_RESTORE, False, ctypes.byref(mii))
                                self.user32.SetMenuDefaultItem(menu, win32con.SC_CLOSE, False)
                            
                            winrect = self.get_window_rect(self.hwnd)
                            
                            cmd = self.user32.TrackPopupMenu(menu, (win32con.TPM_RIGHTBUTTON | win32con.TPM_NONOTIFY | win32con.TPM_RETURNCMD),
                                # When the window is maximized, the pop-up menu activated by "Alt + Space" invades the other monitor.
                                # To fix this, move the window a bit more to the left.
                                winrect.left + 8 if wp.showCmd == win32con.SW_SHOWMAXIMIZED else 0,
                                winrect.top, None, self.hwnd, None)

                            if cmd:
                                win32api.PostMessage(self.hwnd, win32con.WM_SYSCOMMAND, cmd, 0);    
                        return True
                    return False
                case _:
                    pass;    
        elif eventType == "windows_dispatcher_MSG":
            pass
        return False
    
    
    @Slot()
    def onMinimizeButtonClicked(self):
        win32api.SendMessage(self.hwnd, win32con.WM_SYSCOMMAND, win32con.SC_MINIMIZE, 0); 
    
    @Slot()
    def onMaximizeButtonClicked(self):
        maximized = bool(self.quick_window.property("maximized"))
        wparam = win32con.SC_RESTORE if maximized else win32con.SC_MAXIMIZE
        win32api.SendMessage(self.hwnd, win32con.WM_SYSCOMMAND, wparam)
    
    @Slot()
    def onCloseButtonClicked(self):
        win32api.SendMessage(self.hwnd, win32con.WM_CLOSE, 0, 0)