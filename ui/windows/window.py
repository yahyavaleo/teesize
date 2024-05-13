import os

from PyQt5 import QtCore, QtGui, QtWidgets
from PIL import Image, ImageDraw, ImageFont

from ui.windows.mainwindow import MainWindow
from ui.windows.sidebar import Sidebar


class Window(MainWindow):
    def __init__(self, true_width, true_height, pixeltoinch, margin):
        super().__init__(width=1100, height=800)

        self.initialize_sidebar()
        self.initialize_parameters(true_width, true_height, pixeltoinch, margin)
        self.initialize_settings()

        self.settings_btn.clicked.connect(self.toggle_sidebar)
        self.start_btn.clicked.connect(self.start)
        self.clear_btn.clicked.connect(self.clear_sizes)
        self.sidebar.save_btn.clicked.connect(self.save_settings)
        self.sidebar.reset_btn.clicked.connect(self.reset_settings)

        self.clear_sizes()

    def start(self, _) -> None:
        raise NotImplementedError

    def update_sizes(self, sizes):
        image = Image.open(os.path.join("ui", "assets", "template.png"))
        I1 = ImageDraw.Draw(image)
        font = ImageFont.truetype(os.path.join("ui", "fonts", "JetBrainsMono-Medium.ttf"), 33)

        for name, xy in self.size_positions.items():
            I1.text(xy, f"{round(sizes[name])}in/{round(sizes[name]*2.54)}cm", font=font, fill=(0, 0, 0))

        image.save(os.path.join("ui", "assets", "cache.png"))
        self.diagram.setPixmap(QtGui.QPixmap(os.path.join("ui", "assets", "cache.png")))

        for name, entry in self.table.items():
            entry.setText(f'{round(sizes[name])}"')

    def clear_sizes(self):
        image = Image.open(os.path.join("ui", "assets", "template.png"))
        I1 = ImageDraw.Draw(image)
        font = ImageFont.truetype(os.path.join("ui", "fonts", "JetBrainsMono-Medium.ttf"), 35)

        for name, xy in self.name_positions.items():
            I1.text(xy, f"{name}", font=font, fill=(0, 0, 0))

        image.save(os.path.join("ui", "assets", "cache.png"))
        self.diagram.setPixmap(QtGui.QPixmap(os.path.join("ui", "assets", "cache.png")))

        for entry in self.table.values():
            entry.setText("—")

    def initialize_parameters(self, true_width, true_height, pixeltoinch, margin):
        self.default_true_width = true_width
        self.default_true_height = true_height
        self.default_pixeltoinch = pixeltoinch
        self.margin = margin

        self.true_width = self.default_true_width
        self.true_height = self.default_true_height
        self.pixeltoinch = self.default_pixeltoinch

        self.table = {
            "chest": self.chest_size,
            "waist": self.waist_size,
            "hem": self.hem_size,
            "shoulder": self.shoulder_size,
            "length": self.length_size,
            "neck": self.neck_size,
            "sleeve": self.sleeve_size,
            "cuff": self.cuff_size,
        }

        self.name_positions = {
            "chest": (354, 365),
            "waist": (355, 638),
            "hem": (380, 887),
            "shoulder": (175, 116),
            "length": (507, 525),
            "neck": (448, 38),
            "sleeve": (768, 117),
            "cuff": (795, 357),
        }

        self.size_positions = {
            "chest": (316, 368),
            "waist": (315, 639),
            "hem": (317, 889),
            "shoulder": (170, 118),
            "length": (482, 528),
            "neck": (410, 39),
            "sleeve": (760, 118),
            "cuff": (760, 359),
        }

    def initialize_settings(self):
        self.sidebar.trueheight_field.setText(str(self.default_true_height))
        self.sidebar.truewidth_field.setText(str(self.default_true_width))
        self.sidebar.pixeltoinch_field.setText(str(self.default_pixeltoinch))

    def save_settings(self):
        try:
            self.true_height = float(self.sidebar.trueheight_field.text().strip())
        except ValueError:
            self.sidebar.trueheight_field.setText(str(self.default_true_height))
            self.true_height = self.default_true_height

        try:
            self.true_width = float(self.sidebar.truewidth_field.text().strip())
        except ValueError:
            self.sidebar.truewidth_field.setText(str(self.default_true_width))
            self.true_width = self.default_true_width

        try:
            self.pixeltoinch = float(self.sidebar.pixeltoinch_field.text().strip())
        except ValueError:
            self.sidebar.pixeltoinch_field.setText(str(self.default_pixeltoinch))
            self.pixeltoinch = self.default_pixeltoinch

        self.toggle_sidebar()

    def reset_settings(self):
        self.sidebar.trueheight_field.setText(str(self.default_true_height))
        self.sidebar.truewidth_field.setText(str(self.default_true_width))
        self.sidebar.pixeltoinch_field.setText(str(self.default_pixeltoinch))

        self.true_height = self.default_true_height
        self.true_width = self.default_true_width
        self.pixeltoinch = self.default_pixeltoinch

    def initialize_sidebar(self):
        self.sidebar = Sidebar(self)
        self.sidebar_width = 450
        self.sidebar_hidden = True
        self.sidebar_offset = self.header.height()
        self.sidebar_margin = 2
        self.sidebar.setGeometry(self.body.width(), 0, 0, self.body.height() + self.sidebar_margin * 2)

    def toggle_sidebar(self):
        if self.sidebar_hidden:
            self.show_sidebar()
        else:
            self.hide_sidebar()

    def show_sidebar(self):
        self.sidebar_hidden = False
        self.animation = QtCore.QPropertyAnimation(self.sidebar, b"geometry")
        self.animation.setDuration(250)
        self.animation.setStartValue(
            QtCore.QRect(
                self.body.width(),
                self.sidebar_offset - self.sidebar_margin,
                0,
                self.body.height() + self.sidebar_margin * 2,
            )
        )
        self.animation.setEndValue(
            QtCore.QRect(
                self.body.width() - self.sidebar_width,
                self.sidebar_offset - self.sidebar_margin,
                self.sidebar_width,
                self.body.height() + self.sidebar_margin * 2,
            )
        )
        self.animation.setEasingCurve(QtCore.QEasingCurve.OutQuad)
        self.animation.start()

    def hide_sidebar(self):
        self.sidebar_hidden = True
        self.animation = QtCore.QPropertyAnimation(self.sidebar, b"geometry")
        self.animation.setDuration(250)
        self.animation.setStartValue(
            QtCore.QRect(
                self.body.width() - self.sidebar_width,
                self.sidebar_offset - self.sidebar_margin,
                self.sidebar_width,
                self.body.height() + self.sidebar_margin * 2,
            )
        )
        self.animation.setEndValue(
            QtCore.QRect(
                self.body.width(),
                self.sidebar_offset - self.sidebar_margin,
                0,
                self.body.height() + self.sidebar_margin * 2,
            )
        )
        self.animation.setEasingCurve(QtCore.QEasingCurve.OutQuad)
        self.animation.start()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self.sidebar_hidden:
            self.sidebar.setGeometry(
                self.body.width() - self.sidebar_width,
                self.sidebar_offset - self.sidebar_margin,
                self.sidebar_width,
                self.body.height() + self.sidebar_margin * 2,
            )

    def get_image_file(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Image", ".", "Images (*.png *.jpg *.jpeg)")
        return filename

    def disable(self, flag):
        self.start_btn.setDisabled(flag)
        self.clear_btn.setDisabled(flag)


def get_app(args):
    app = QtWidgets.QApplication(args)
    return app


def wait(func):
    def wrapper(self, *args, **kwargs):
        self.disable(True)
        try:
            ret = func(self, *args, **kwargs)
        finally:
            self.disable(False)
        return ret

    return wrapper
