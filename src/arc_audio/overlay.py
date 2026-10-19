import math
import time

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from .dataset import CLASS_NAMES
from .windows import apply_overlay_styles


class Overlay(QWidget):
    def __init__(self, settings):
        super().__init__(None, Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint
                         | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.WindowTransparentForInput
                         | Qt.WindowType.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.settings = settings
        self.native_error = ""
        self.indicators = []
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(16)

    def showEvent(self, event):
        super().showEvent(event)
        try:
            apply_overlay_styles(int(self.winId()))
        except OSError as exc:
            self.native_error = f"Windows overlay setup failed: {exc}. Overlay hidden; restart the app."
            QTimer.singleShot(0, self.hide)

    def target(self, screen):
        # QScreen geometry uses device-independent pixels, including negative origins.
        # Do not multiply these coordinates by devicePixelRatio a second time.
        self.winId()
        self.windowHandle().setScreen(screen)
        self.setGeometry(screen.geometry())

    def add(self, event):
        # Renew the same cue rather than stacking alerts for one ongoing sound.
        # Multiple cue types are not a claim about the number of sound sources.
        key = (self._lane(event.direction), event.label)
        self.indicators = [(e, born) for e, born in self.indicators
                           if (self._lane(e.direction), e.label) != key]
        self.indicators.append((event, time.monotonic()))
        self.indicators = self.indicators[-12:]
        self.update()

    def clear(self):
        self.indicators.clear()
        self.update()

    def tick(self):
        now = time.monotonic()
        self.indicators = [(e, born) for e, born in self.indicators if now - born < self.settings.lifetime]
        if self.isVisible():
            self.update()

    def paintEvent(self, _event):
        if not self.indicators:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        # Two broad side sectors, not a 360-degree bearing estimate. Balanced
        # audio is drawn separately inside the ring with an explicit unknown label.
        scale = min(self.settings.ui_scale, self.width() / 850, self.height() / 480)
        cx, cy = self.width() / 2, self.height() / 2
        radius = 150 * scale
        groups = {"left": [], "unknown": [], "right": []}
        for event, born in self.indicators:
            groups[self._lane(event.direction)].append((event, born))
        now = time.monotonic()
        colors = {"sound activity": "#65e6ca", "transient": "#ffcf75",
                  "footsteps": "#81b8ff", "gunshot": "#ff8191", "reload": "#c8a0ff",
                  "landing": "#ffc597", "snap_hook": "#91dfff", "zipline": "#9fe39f",
                  "climb_vault": "#e6d692", "healing": "#fa9ec6"}
        for lane, entries in groups.items():
            if not entries:
                continue
            # Keep each side compact when several classes coexist.
            entries = entries[-2:]
            if lane != "unknown":
                event, born = entries[-1]
                age = max(0, now - born)
                color = QColor(colors.get(event.label, "#ffffff"))
                painter.setOpacity(self.settings.opacity * self._envelope(age))
                self._sector(painter, cx, cy, radius, lane, color, scale, age)
            for index, (event, born) in enumerate(entries):
                age = max(0, now - born)
                painter.setOpacity(self.settings.opacity * self._envelope(age))
                color = QColor(colors.get(event.label, "#ffffff"))
                if lane == "unknown":
                    # The center cue has no compass arc, so it cannot imply rear/front.
                    x, y = cx, cy + (65 + index * 80) * scale
                    self._unknown_cue(painter, event, x, y, scale, color)
                else:
                    sign = -1 if lane == "left" else 1
                    x = cx + sign * (radius + 48 * scale)
                    y = cy + (index - (len(entries) - 1) / 2) * 112 * scale
                    self._side_cue(painter, event, x, y, scale, color, lane)

    @staticmethod
    def _lane(direction):
        return direction if direction in ("left", "right") else "unknown"

    def _envelope(self, age):
        attack = min(1.0, age / 0.09)
        fade_time = min(0.45, self.settings.lifetime / 2)
        fade = min(1.0, max(0.0, (self.settings.lifetime - age) / fade_time))
        return attack * fade

    @staticmethod
    def _sector(p, cx, cy, radius, lane, color, scale, age):
        p.setBrush(Qt.BrushStyle.NoBrush)
        start = 140 if lane == "left" else -40
        # Quick outward pulse on arrival, then settle. No compass ticks/angles.
        pulse = math.sin(min(age / 0.28, 1.0) * math.pi) * 7 * scale
        radius += pulse
        arc = QRectF(cx - radius, cy - radius, radius * 2, radius * 2)
        glow = QColor(color)
        glow.setAlpha(45)
        p.setPen(QPen(glow, 17 * scale, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawArc(arc, start * 16, 80 * 16)
        p.setPen(QPen(QColor(8, 16, 24, 220), 7 * scale, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawArc(arc, start * 16, 80 * 16)
        p.setPen(QPen(color, 3.5 * scale, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawArc(arc, start * 16, 80 * 16)

    def _side_cue(self, p, event, x, y, s, color, lane):
        self._badge(p, event.label, x, y, s, color)
        self._text(p, QRectF(x - 115 * s, y + 32 * s, 230 * s, 22 * s),
                   CLASS_NAMES.get(event.label, event.label).upper(), 11, s, QColor("#ffffff"), bold=True)
        origin = "HEURISTIC" if event.backend == "baseline" else "MODEL CLIP"
        self._text(p, QRectF(x - 115 * s, y + 55 * s, 230 * s, 18 * s),
                   f"{lane.upper()} BALANCE · {origin}", 8, s, color)

    def _unknown_cue(self, p, event, x, y, s, color):
        rect = QRectF(x - 120 * s, y - 25 * s, 240 * s, 64 * s)
        p.setPen(QPen(QColor(color.red(), color.green(), color.blue(), 120), s))
        p.setBrush(QColor(10, 18, 27, 215))
        p.drawRoundedRect(rect, 14 * s, 14 * s)
        self._icon(p, event.label, x - 94 * s, y - 2 * s, .8 * s, color)
        self._text(p, QRectF(x - 72 * s, y - 20 * s, 184 * s, 23 * s),
                   CLASS_NAMES.get(event.label, event.label).upper(), 10, s, QColor("#ffffff"), bold=True)
        self._text(p, QRectF(x - 114 * s, y + 11 * s, 228 * s, 20 * s),
                   "DIRECTION UNKNOWN", 8, s, color)

    def _badge(self, p, label, x, y, s, color):
        p.setPen(QPen(color, 1.5 * s))
        p.setBrush(QColor(10, 18, 27, 225))
        p.drawEllipse(QPointF(x, y), 25 * s, 25 * s)
        self._icon(p, label, x, y, s, color)

    @staticmethod
    def _text(p, rect, text, size, scale, color, bold=False):
        font = QFont("Segoe UI")
        font.setPixelSize(max(7, round(size * scale)))
        font.setWeight(QFont.Weight.DemiBold if bold else QFont.Weight.Normal)
        p.setFont(font)
        # A small outline holds up over a bright scene without a large card.
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            p.setPen(QColor(0, 0, 0, 210))
            p.drawText(rect.translated(dx, dy), Qt.AlignmentFlag.AlignCenter, text)
        p.setPen(color)
        p.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    @staticmethod
    def _icon(p, label, x, y, s, color):
        p.setPen(QPen(color, 2 * s))
        p.setBrush(Qt.BrushStyle.NoBrush)
        if label == "transient":
            p.drawPolygon(QPolygonF([QPointF(x, y - 11 * s), QPointF(x + 10 * s, y),
                                     QPointF(x, y + 11 * s), QPointF(x - 10 * s, y)]))
        elif label == "footsteps":
            p.drawEllipse(QRectF(x - 10 * s, y - 10 * s, 7 * s, 13 * s))
            p.drawEllipse(QRectF(x + 2 * s, y - 2 * s, 7 * s, 13 * s))
        elif label == "gunshot":
            for dx, dy in [(12, 0), (0, 12), (8, 8), (8, -8)]:
                p.drawLine(QPointF(x - dx * s, y - dy * s), QPointF(x + dx * s, y + dy * s))
        elif label == "reload":
            p.drawRoundedRect(QRectF(x - 5 * s, y - 7 * s, 10 * s, 18 * s), 2 * s, 2 * s)
            p.drawArc(QRectF(x - 13 * s, y - 14 * s, 26 * s, 26 * s), 15 * 16, 130 * 16)
            p.drawLine(QPointF(x + 12 * s, y - 10 * s), QPointF(x + 12 * s, y - 3 * s))
            p.drawLine(QPointF(x + 12 * s, y - 3 * s), QPointF(x + 6 * s, y - 6 * s))
        elif label == "landing":
            p.drawLine(QPointF(x, y - 12 * s), QPointF(x, y + 5 * s))
            p.drawLine(QPointF(x, y + 5 * s), QPointF(x - 6 * s, y - 1 * s))
            p.drawLine(QPointF(x, y + 5 * s), QPointF(x + 6 * s, y - 1 * s))
            p.drawLine(QPointF(x - 12 * s, y + 11 * s), QPointF(x + 12 * s, y + 11 * s))
        elif label == "snap_hook":
            p.drawLine(QPointF(x + 5 * s, y - 13 * s), QPointF(x + 5 * s, y + 3 * s))
            p.drawArc(QRectF(x - 9 * s, y - 5 * s, 14 * s, 18 * s), 180 * 16, 180 * 16)
        elif label == "zipline":
            p.drawLine(QPointF(x - 14 * s, y - 9 * s), QPointF(x + 14 * s, y - 3 * s))
            p.drawLine(QPointF(x, y - 6 * s), QPointF(x, y + 5 * s))
            p.drawEllipse(QPointF(x, y + 8 * s), 4 * s, 4 * s)
        elif label == "climb_vault":
            points = [QPointF(x - 12 * s, y + 10 * s), QPointF(x - 4 * s, y + 10 * s),
                      QPointF(x - 4 * s, y + 2 * s), QPointF(x + 4 * s, y + 2 * s),
                      QPointF(x + 4 * s, y - 7 * s), QPointF(x + 12 * s, y - 7 * s)]
            p.drawPolyline(QPolygonF(points))
        elif label == "healing":
            p.drawLine(QPointF(x - 11 * s, y), QPointF(x + 11 * s, y))
            p.drawLine(QPointF(x, y - 11 * s), QPointF(x, y + 11 * s))
        else:
            for dx, height in [(-8, 5), (0, 11), (8, 7)]:
                p.drawLine(QPointF(x + dx * s, y - height * s), QPointF(x + dx * s, y + height * s))
