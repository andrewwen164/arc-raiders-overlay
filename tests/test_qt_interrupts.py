import signal

import pytest


@pytest.fixture
def qapp(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_interrupt_defers_shutdown_and_restores_handlers(qapp):
    from arc_audio.qt_interrupts import ConsoleInterrupts
    numbers = [getattr(signal, name) for name in ("SIGINT", "SIGBREAK") if hasattr(signal, name)]
    previous = {number: signal.getsignal(number) for number in numbers}
    closed = []
    with ConsoleInterrupts(lambda: closed.append(True)) as bridge:
        signal.raise_signal(signal.SIGINT)
        assert bridge.interrupted and not closed
        bridge.poll()
        assert closed == [True]
        bridge.poll()
        assert closed == [True]
        signal.raise_signal(numbers[-1])
        bridge.poll()
        assert len(closed) == 2
    assert not bridge.timer.isActive()
    assert {number: signal.getsignal(number) for number in numbers} == previous


def test_qt_loop_exits_without_keyboardinterrupt_traceback(qapp):
    from PySide6.QtCore import QTimer

    from arc_audio.qt_interrupts import run_event_loop
    interrupt = QTimer()
    interrupt.setSingleShot(True)
    interrupt.timeout.connect(lambda: signal.raise_signal(signal.SIGINT))
    interrupt.start(20)
    watchdog = QTimer()
    watchdog.setSingleShot(True)
    watchdog.timeout.connect(qapp.quit)
    watchdog.start(3000)
    closed = []

    def close():
        closed.append(True)
        qapp.quit()

    try:
        assert run_event_loop(qapp, close) == 130
        assert closed == [True]
    finally:
        interrupt.stop()
        watchdog.stop()


def test_interrupt_cancels_modal_dialog_before_normal_close(qapp):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QDialog

    from arc_audio.qt_interrupts import ConsoleInterrupts
    dialog = QDialog()
    dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
    dialog.show()
    qapp.processEvents()
    closed = []
    with ConsoleInterrupts(lambda: closed.append(True)) as bridge:
        signal.raise_signal(signal.SIGINT)
        bridge.poll()
    assert closed and not dialog.isVisible()


def test_handlers_are_restored_on_exception(qapp):
    from arc_audio.qt_interrupts import ConsoleInterrupts
    previous = signal.getsignal(signal.SIGINT)
    with pytest.raises(RuntimeError, match="fixture"):
        with ConsoleInterrupts(lambda: None):
            raise RuntimeError("fixture")
    assert signal.getsignal(signal.SIGINT) == previous
