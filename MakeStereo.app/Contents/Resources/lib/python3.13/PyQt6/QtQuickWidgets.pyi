# The PEP 484 type hints stub file for the QtQuickWidgets module.
#
# Generated by SIP 6.12.0
#
# Copyright (c) 2025 Riverbank Computing Limited <info@riverbankcomputing.com>
# 
# This file is part of PyQt6.
# 
# This file may be used under the terms of the GNU General Public License
# version 3.0 as published by the Free Software Foundation and appearing in
# the file LICENSE included in the packaging of this file.  Please review the
# following information to ensure the GNU General Public License version 3.0
# requirements will be met: http://www.gnu.org/copyleft/gpl.html.
# 
# If you do not wish to use this file under the terms of the GPL version 3.0
# then you may purchase a commercial license.  For more information contact
# info@riverbankcomputing.com.
# 
# This file is provided AS IS with NO WARRANTY OF ANY KIND, INCLUDING THE
# WARRANTY OF DESIGN, MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.


import collections, re, typing, enum

try:
    from warnings import deprecated
except ImportError:
    pass

import PyQt6.sip

from PyQt6 import QtCore
from PyQt6 import QtGui
from PyQt6 import QtNetwork
from PyQt6 import QtQml
from PyQt6 import QtQuick
from PyQt6 import QtWidgets

# Support for QDate, QDateTime and QTime.
import datetime

# Convenient type aliases.
PYQT_SIGNAL = typing.Union[QtCore.pyqtSignal, QtCore.pyqtBoundSignal]
PYQT_SLOT = typing.Union[collections.abc.Callable[..., Any], QtCore.pyqtBoundSignal]


class QQuickWidget(QtWidgets.QWidget):

    class Status(enum.Enum):
        Null = ... # type: QQuickWidget.Status
        Ready = ... # type: QQuickWidget.Status
        Loading = ... # type: QQuickWidget.Status
        Error = ... # type: QQuickWidget.Status

    class ResizeMode(enum.Enum):
        SizeViewToRootObject = ... # type: QQuickWidget.ResizeMode
        SizeRootObjectToView = ... # type: QQuickWidget.ResizeMode

    @typing.overload
    def __init__(self, parent: typing.Optional[QtWidgets.QWidget] = ...) -> None: ...
    @typing.overload
    def __init__(self, engine: typing.Optional[QtQml.QQmlEngine], parent: typing.Optional[QtWidgets.QWidget]) -> None: ...
    @typing.overload
    def __init__(self, source: QtCore.QUrl, parent: typing.Optional[QtWidgets.QWidget] = ...) -> None: ...
    @typing.overload
    def __init__(self, uri: typing.Union[typing.Union[QtCore.QByteArray, bytes, bytearray, memoryview], typing.Optional[str]], typeName: typing.Union[typing.Union[QtCore.QByteArray, bytes, bytearray, memoryview], typing.Optional[str]], parent: typing.Optional[QtWidgets.QWidget] = ...) -> None: ...

    def loadFromModule(self, uri: typing.Union[typing.Union[QtCore.QByteArray, bytes, bytearray, memoryview], typing.Optional[str]], typeName: typing.Union[typing.Union[QtCore.QByteArray, bytes, bytearray, memoryview], typing.Optional[str]]) -> None: ...
    def setInitialProperties(self, initialProperties: dict[typing.Optional[str], typing.Any]) -> None: ...
    def quickWindow(self) -> typing.Optional[QtQuick.QQuickWindow]: ...
    def setClearColor(self, color: typing.Union[QtGui.QColor, QtCore.Qt.GlobalColor, int]) -> None: ...
    def grabFramebuffer(self) -> QtGui.QImage: ...
    def focusNextPrevChild(self, next: bool) -> bool: ...
    def paintEvent(self, event: typing.Optional[QtGui.QPaintEvent]) -> None: ...
    def dropEvent(self, a0: typing.Optional[QtGui.QDropEvent]) -> None: ...
    def dragLeaveEvent(self, a0: typing.Optional[QtGui.QDragLeaveEvent]) -> None: ...
    def dragMoveEvent(self, a0: typing.Optional[QtGui.QDragMoveEvent]) -> None: ...
    def dragEnterEvent(self, a0: typing.Optional[QtGui.QDragEnterEvent]) -> None: ...
    def focusOutEvent(self, event: typing.Optional[QtGui.QFocusEvent]) -> None: ...
    def focusInEvent(self, event: typing.Optional[QtGui.QFocusEvent]) -> None: ...
    def event(self, a0: typing.Optional[QtCore.QEvent]) -> bool: ...
    def wheelEvent(self, a0: typing.Optional[QtGui.QWheelEvent]) -> None: ...
    def hideEvent(self, a0: typing.Optional[QtGui.QHideEvent]) -> None: ...
    def showEvent(self, a0: typing.Optional[QtGui.QShowEvent]) -> None: ...
    def mouseDoubleClickEvent(self, a0: typing.Optional[QtGui.QMouseEvent]) -> None: ...
    def mouseMoveEvent(self, a0: typing.Optional[QtGui.QMouseEvent]) -> None: ...
    def mouseReleaseEvent(self, a0: typing.Optional[QtGui.QMouseEvent]) -> None: ...
    def mousePressEvent(self, a0: typing.Optional[QtGui.QMouseEvent]) -> None: ...
    def keyReleaseEvent(self, a0: typing.Optional[QtGui.QKeyEvent]) -> None: ...
    def keyPressEvent(self, a0: typing.Optional[QtGui.QKeyEvent]) -> None: ...
    def timerEvent(self, a0: typing.Optional[QtCore.QTimerEvent]) -> None: ...
    def resizeEvent(self, a0: typing.Optional[QtGui.QResizeEvent]) -> None: ...
    sceneGraphError: typing.ClassVar[QtCore.pyqtSignal]
    statusChanged: typing.ClassVar[QtCore.pyqtSignal]
    def setSource(self, a0: QtCore.QUrl) -> None: ...
    def format(self) -> QtGui.QSurfaceFormat: ...
    def setFormat(self, format: QtGui.QSurfaceFormat) -> None: ...
    def initialSize(self) -> QtCore.QSize: ...
    def sizeHint(self) -> QtCore.QSize: ...
    def errors(self) -> list[QtQml.QQmlError]: ...
    def status(self) -> 'QQuickWidget.Status': ...
    def setResizeMode(self, a0: 'QQuickWidget.ResizeMode') -> None: ...
    def resizeMode(self) -> 'QQuickWidget.ResizeMode': ...
    def rootObject(self) -> typing.Optional[QtQuick.QQuickItem]: ...
    def rootContext(self) -> typing.Optional[QtQml.QQmlContext]: ...
    def engine(self) -> typing.Optional[QtQml.QQmlEngine]: ...
    def source(self) -> QtCore.QUrl: ...
