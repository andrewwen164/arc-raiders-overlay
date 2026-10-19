"""Render a layout preview from explicit UI fixtures, without capture/inference."""

import time
from pathlib import Path

from PySide6.QtCore import QPoint
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QApplication

from arc_audio.dataset import CLASS_NAMES, TARGET_CLASSES
from arc_audio.events import Detection, utc_now
from arc_audio.overlay import Overlay
from arc_audio.settings import Settings


def main():
    app = QApplication([])
    overlay = Overlay(Settings())
    overlay.resize(1280, 720)
    # Deliberate layout fixtures, not predictions or a recording of game audio.
    for label, direction in [("transient", "left"), ("sound activity", "right"),
                             ("sound activity", "unknown")]:
        overlay.add(Detection(utc_now(), 0, label, .8, direction, "baseline"))
    overlay.indicators = [(event, time.monotonic() - .15) for event, _born in overlay.indicators]
    image = QImage(overlay.size(), QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(QColor("#1a2936"))
    painter = QPainter(image)
    overlay.render(painter, QPoint())
    painter.end()
    output = Path(__file__).resolve().parents[1] / "artifacts" / "compass-preview.png"
    output.parent.mkdir(exist_ok=True)
    if not image.save(str(output)):
        raise RuntimeError(f"Cannot save {output}")
    print(f"UI fixtures only: {output}")
    catalog = QImage(1000, 360, QImage.Format.Format_ARGB32_Premultiplied)
    catalog.fill(QColor("#1a2936"))
    painter = QPainter(catalog)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    for index, label in enumerate(TARGET_CLASSES):
        from PySide6.QtCore import QRectF
        x, y = 125 + index % 4 * 250, 60 + index // 4 * 180
        overlay._badge(painter, label, x, y, 1, QColor("#91dfff"))
        overlay._text(painter, QRectF(x - 115, y + 35, 230, 32), CLASS_NAMES[label], 12, 1, QColor("#ffffff"))
    painter.end()
    icon_path = output.with_name("sound-icons.png")
    if not catalog.save(str(icon_path)):
        raise RuntimeError(f"Cannot save {icon_path}")
    print(f"Supported icon layout fixtures: {icon_path}")
    overlay.close()
    app.quit()


if __name__ == "__main__":
    main()
