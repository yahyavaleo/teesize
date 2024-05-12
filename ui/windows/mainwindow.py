import os
from PyQt5 import QtCore, QtGui, QtWidgets


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, width, height):
        super().__init__()
        self.setObjectName("MainWindow")
        self.resize(width, height)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(os.path.join("ui", "assets", "logo.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setWindowIcon(icon)

        JetBrainsMonoMedium = self.loadFont(os.path.join("ui", "fonts", "JetBrainsMono-Medium.ttf"))
        JetBrainsMonoBold = self.loadFont(os.path.join("ui", "fonts", "JetBrainsMono-Bold.ttf"))
        JetBrainsMonoExtraBold = self.loadFont(os.path.join("ui", "fonts", "JetBrainsMono-ExtraBold.ttf"))

        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setStyleSheet("QWidget {\n" "    background-color: #efefef;\n" "}")
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.header = QtWidgets.QWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.header.sizePolicy().hasHeightForWidth())
        self.header.setSizePolicy(sizePolicy)
        self.header.setMinimumSize(QtCore.QSize(0, 70))
        self.header.setStyleSheet(
            "QWidget {    \n" "    border-top: 2px solid black;\n" "    border-bottom: 2px solid black;\n" "}"
        )
        self.header.setObjectName("header")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.header)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Minimum
        )
        self.horizontalLayout.addItem(spacerItem)
        self.label = QtWidgets.QLabel(self.header)
        self.label.setText("")
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.logo = QtWidgets.QLabel(self.header)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.logo.sizePolicy().hasHeightForWidth())
        self.logo.setSizePolicy(sizePolicy)
        self.logo.setMinimumSize(QtCore.QSize(850, 0))
        font = QtGui.QFont(JetBrainsMonoExtraBold)
        font.setPointSize(18)
        font.setBold(False)
        font.setWeight(100)
        self.logo.setFont(font)
        self.logo.setObjectName("logo")
        self.horizontalLayout.addWidget(self.logo)
        spacerItem1 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Minimum
        )
        self.horizontalLayout.addItem(spacerItem1)
        self.settings_btn = QtWidgets.QPushButton(self.header)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.settings_btn.sizePolicy().hasHeightForWidth())
        self.settings_btn.setSizePolicy(sizePolicy)
        self.settings_btn.setMinimumSize(QtCore.QSize(50, 50))
        self.settings_btn.setMaximumSize(QtCore.QSize(50, 50))
        self.settings_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        self.settings_btn.setStyleSheet(
            "QPushButton{\n"
            "    color: black;\n"
            "    background-color: #f3f5fc;\n"
            "    padding: 8px;\n"
            "    border: 2px solid black;\n"
            "}\n"
            "\n"
            "QPushButton:hover{\n"
            "    background-color: #c3c5cc; \n"
            "}\n"
            "\n"
            "QPushButton:pressed{\n"
            "    color: white;\n"
            "    background-color: #a388ee;\n"
            "}"
        )
        self.settings_btn.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(
            QtGui.QPixmap(os.path.join("ui", "assets", "settings.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off
        )
        self.settings_btn.setIcon(icon1)
        self.settings_btn.setIconSize(QtCore.QSize(25, 25))
        self.settings_btn.setFlat(True)
        self.settings_btn.setObjectName("settings_btn")
        self.horizontalLayout.addWidget(self.settings_btn)
        spacerItem2 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Minimum
        )
        self.horizontalLayout.addItem(spacerItem2)
        self.verticalLayout.addWidget(self.header)
        self.body = QtWidgets.QWidget(self.centralwidget)
        self.body.setObjectName("body")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.body)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        spacerItem3 = QtWidgets.QSpacerItem(
            20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.MinimumExpanding
        )
        self.verticalLayout_2.addItem(spacerItem3)
        self.tcontainer = QtWidgets.QWidget(self.body)
        self.tcontainer.setObjectName("tcontainer")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.tcontainer)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setSpacing(0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.container = QtWidgets.QWidget(self.tcontainer)
        self.container.setMinimumSize(QtCore.QSize(0, 0))
        self.container.setObjectName("container")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.container)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setSpacing(0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        spacerItem4 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Minimum
        )
        self.horizontalLayout_3.addItem(spacerItem4)
        self.diagram = QtWidgets.QLabel(self.container)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.diagram.sizePolicy().hasHeightForWidth())
        self.diagram.setSizePolicy(sizePolicy)
        self.diagram.setMinimumSize(QtCore.QSize(500, 545))
        self.diagram.setMaximumSize(QtCore.QSize(500, 545))
        self.diagram.setStyleSheet("QLabel {\n" "    border: 2px solid black;\n" "}")
        self.diagram.setText("")
        self.diagram.setPixmap(QtGui.QPixmap(os.path.join("ui", "assets", "template.png")))
        self.diagram.setScaledContents(True)
        self.diagram.setObjectName("diagram")
        self.horizontalLayout_3.addWidget(self.diagram)
        spacerItem5 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Minimum
        )
        self.horizontalLayout_3.addItem(spacerItem5)
        self.rcontainer = QtWidgets.QWidget(self.container)
        self.rcontainer.setMinimumSize(QtCore.QSize(0, 0))
        self.rcontainer.setObjectName("rcontainer")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.rcontainer)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setSpacing(0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.table = QtWidgets.QWidget(self.rcontainer)
        self.table.setMinimumSize(QtCore.QSize(0, 0))
        self.table.setStyleSheet("QWidget {\n" "    background-color: white;\n" "    border: 2px solid black;\n" "}")
        self.table.setObjectName("table")
        self.gridLayout = QtWidgets.QGridLayout(self.table)
        self.gridLayout.setContentsMargins(2, 2, 2, 2)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setObjectName("gridLayout")
        self.hem_row = QtWidgets.QLabel(self.table)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.hem_row.sizePolicy().hasHeightForWidth())
        self.hem_row.setSizePolicy(sizePolicy)
        self.hem_row.setMinimumSize(QtCore.QSize(200, 45))
        font = QtGui.QFont(JetBrainsMonoMedium)
        font.setPointSize(11)
        font.setWeight(55)
        self.hem_row.setFont(font)
        self.hem_row.setStyleSheet("QLabel {\n" "    border: none;\n" "    border-top: 1px solid black;\n" "}")
        self.hem_row.setAlignment(QtCore.Qt.AlignCenter)
        self.hem_row.setObjectName("hem_row")
        self.gridLayout.addWidget(self.hem_row, 3, 0, 1, 1)
        self.shoulder_size = QtWidgets.QLabel(self.table)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.shoulder_size.sizePolicy().hasHeightForWidth())
        self.shoulder_size.setSizePolicy(sizePolicy)
        self.shoulder_size.setMinimumSize(QtCore.QSize(200, 45))
        font = QtGui.QFont(JetBrainsMonoMedium)
        font.setPointSize(11)
        font.setWeight(55)
        self.shoulder_size.setFont(font)
        self.shoulder_size.setStyleSheet("QLabel {\n" "    border: none;\n" "    border-top: 1px solid black;\n" "}")
        self.shoulder_size.setAlignment(QtCore.Qt.AlignCenter)
        self.shoulder_size.setObjectName("shoulder_size")
        self.gridLayout.addWidget(self.shoulder_size, 4, 1, 1, 1)
        self.length_size = QtWidgets.QLabel(self.table)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.length_size.sizePolicy().hasHeightForWidth())
        self.length_size.setSizePolicy(sizePolicy)
        self.length_size.setMinimumSize(QtCore.QSize(200, 45))
        font = QtGui.QFont(JetBrainsMonoMedium)
        font.setPointSize(11)
        font.setWeight(55)
        self.length_size.setFont(font)
        self.length_size.setStyleSheet("QLabel {\n" "    border: none;\n" "    border-top: 1px solid black;\n" "}")
        self.length_size.setAlignment(QtCore.Qt.AlignCenter)
        self.length_size.setObjectName("length_size")
        self.gridLayout.addWidget(self.length_size, 5, 1, 1, 1)
        self.waist_size = QtWidgets.QLabel(self.table)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.waist_size.sizePolicy().hasHeightForWidth())
        self.waist_size.setSizePolicy(sizePolicy)
        self.waist_size.setMinimumSize(QtCore.QSize(200, 45))
        font = QtGui.QFont(JetBrainsMonoMedium)
        font.setPointSize(11)
        font.setWeight(55)
        self.waist_size.setFont(font)
        self.waist_size.setStyleSheet("QLabel {\n" "    border: none;\n" "    border-top: 1px solid black;\n" "}")
        self.waist_size.setAlignment(QtCore.Qt.AlignCenter)
        self.waist_size.setObjectName("waist_size")
        self.gridLayout.addWidget(self.waist_size, 2, 1, 1, 1)
        self.length_row = QtWidgets.QLabel(self.table)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.length_row.sizePolicy().hasHeightForWidth())
        self.length_row.setSizePolicy(sizePolicy)
        self.length_row.setMinimumSize(QtCore.QSize(200, 45))
        font = QtGui.QFont(JetBrainsMonoMedium)
        font.setPointSize(11)
        font.setWeight(55)
        self.length_row.setFont(font)
        self.length_row.setStyleSheet("QLabel {\n" "    border: none;\n" "    border-top: 1px solid black;\n" "}")
        self.length_row.setAlignment(QtCore.Qt.AlignCenter)
        self.length_row.setObjectName("length_row")
        self.gridLayout.addWidget(self.length_row, 5, 0, 1, 1)
        self.neck_row = QtWidgets.QLabel(self.table)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.neck_row.sizePolicy().hasHeightForWidth())
        self.neck_row.setSizePolicy(sizePolicy)
        self.neck_row.setMinimumSize(QtCore.QSize(200, 45))
        font = QtGui.QFont(JetBrainsMonoMedium)
        font.setPointSize(11)
        font.setWeight(55)
        self.neck_row.setFont(font)
        self.neck_row.setStyleSheet("QLabel {\n" "    border: none;\n" "    border-top: 1px solid black;\n" "}")
        self.neck_row.setAlignment(QtCore.Qt.AlignCenter)
        self.neck_row.setObjectName("neck_row")
        self.gridLayout.addWidget(self.neck_row, 6, 0, 1, 1)
        self.chest_size = QtWidgets.QLabel(self.table)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.chest_size.sizePolicy().hasHeightForWidth())
        self.chest_size.setSizePolicy(sizePolicy)
        self.chest_size.setMinimumSize(QtCore.QSize(200, 45))
        font = QtGui.QFont(JetBrainsMonoMedium)
        font.setPointSize(11)
        font.setWeight(55)
        self.chest_size.setFont(font)
        self.chest_size.setStyleSheet("QLabel {\n" "    border: none;\n" "    border-top: 2px solid black;\n" "}")
        self.chest_size.setAlignment(QtCore.Qt.AlignCenter)
        self.chest_size.setObjectName("chest_size")
        self.gridLayout.addWidget(self.chest_size, 1, 1, 1, 1)
        self.chest_row = QtWidgets.QLabel(self.table)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.chest_row.sizePolicy().hasHeightForWidth())
        self.chest_row.setSizePolicy(sizePolicy)
        self.chest_row.setMinimumSize(QtCore.QSize(200, 45))
        font = QtGui.QFont(JetBrainsMonoMedium)
        font.setPointSize(11)
        font.setWeight(55)
        self.chest_row.setFont(font)
        self.chest_row.setStyleSheet("QLabel {\n" "    border: none;\n" "    border-top: 2px solid black;\n" "}")
        self.chest_row.setAlignment(QtCore.Qt.AlignCenter)
        self.chest_row.setObjectName("chest_row")
        self.gridLayout.addWidget(self.chest_row, 1, 0, 1, 1)
        self.waist_row = QtWidgets.QLabel(self.table)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.waist_row.sizePolicy().hasHeightForWidth())
        self.waist_row.setSizePolicy(sizePolicy)
        self.waist_row.setMinimumSize(QtCore.QSize(200, 45))
        font = QtGui.QFont(JetBrainsMonoMedium)
        font.setPointSize(11)
        font.setWeight(55)
        self.waist_row.setFont(font)
        self.waist_row.setStyleSheet("QLabel {\n" "    border: none;\n" "    border-top: 1px solid black;\n" "}")
        self.waist_row.setAlignment(QtCore.Qt.AlignCenter)
        self.waist_row.setObjectName("waist_row")
        self.gridLayout.addWidget(self.waist_row, 2, 0, 1, 1)
        self.cuff_size = QtWidgets.QLabel(self.table)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cuff_size.sizePolicy().hasHeightForWidth())
        self.cuff_size.setSizePolicy(sizePolicy)
        self.cuff_size.setMinimumSize(QtCore.QSize(200, 45))
        font = QtGui.QFont(JetBrainsMonoMedium)
        font.setPointSize(11)
        font.setWeight(55)
        self.cuff_size.setFont(font)
        self.cuff_size.setStyleSheet("QLabel {\n" "    border: none;\n" "    border-top: 1px solid black;\n" "}")
        self.cuff_size.setAlignment(QtCore.Qt.AlignCenter)
        self.cuff_size.setObjectName("cuff_size")
        self.gridLayout.addWidget(self.cuff_size, 8, 1, 1, 1)
        self.cuff_row = QtWidgets.QLabel(self.table)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cuff_row.sizePolicy().hasHeightForWidth())
        self.cuff_row.setSizePolicy(sizePolicy)
        self.cuff_row.setMinimumSize(QtCore.QSize(200, 45))
        font = QtGui.QFont(JetBrainsMonoMedium)
        font.setPointSize(11)
        font.setWeight(55)
        self.cuff_row.setFont(font)
        self.cuff_row.setStyleSheet("QLabel {\n" "    border: none;\n" "    border-top: 1px solid black;\n" "}")
        self.cuff_row.setAlignment(QtCore.Qt.AlignCenter)
        self.cuff_row.setObjectName("cuff_row")
        self.gridLayout.addWidget(self.cuff_row, 8, 0, 1, 1)
        self.neck_size = QtWidgets.QLabel(self.table)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.neck_size.sizePolicy().hasHeightForWidth())
        self.neck_size.setSizePolicy(sizePolicy)
        self.neck_size.setMinimumSize(QtCore.QSize(200, 45))
        font = QtGui.QFont(JetBrainsMonoMedium)
        font.setPointSize(11)
        font.setWeight(55)
        self.neck_size.setFont(font)
        self.neck_size.setStyleSheet("QLabel {\n" "    border: none;\n" "    border-top: 1px solid black;\n" "}")
        self.neck_size.setAlignment(QtCore.Qt.AlignCenter)
        self.neck_size.setObjectName("neck_size")
        self.gridLayout.addWidget(self.neck_size, 6, 1, 1, 1)
        self.sleeve_size = QtWidgets.QLabel(self.table)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sleeve_size.sizePolicy().hasHeightForWidth())
        self.sleeve_size.setSizePolicy(sizePolicy)
        self.sleeve_size.setMinimumSize(QtCore.QSize(200, 45))
        font = QtGui.QFont(JetBrainsMonoMedium)
        font.setPointSize(11)
        font.setWeight(55)
        self.sleeve_size.setFont(font)
        self.sleeve_size.setStyleSheet("QLabel {\n" "    border: none;\n" "    border-top: 1px solid black;\n" "}")
        self.sleeve_size.setAlignment(QtCore.Qt.AlignCenter)
        self.sleeve_size.setObjectName("sleeve_size")
        self.gridLayout.addWidget(self.sleeve_size, 7, 1, 1, 1)
        self.sleeve_row = QtWidgets.QLabel(self.table)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sleeve_row.sizePolicy().hasHeightForWidth())
        self.sleeve_row.setSizePolicy(sizePolicy)
        self.sleeve_row.setMinimumSize(QtCore.QSize(200, 45))
        font = QtGui.QFont(JetBrainsMonoMedium)
        font.setPointSize(11)
        font.setWeight(55)
        self.sleeve_row.setFont(font)
        self.sleeve_row.setStyleSheet("QLabel {\n" "    border: none;\n" "    border-top: 1px solid black;\n" "}")
        self.sleeve_row.setAlignment(QtCore.Qt.AlignCenter)
        self.sleeve_row.setObjectName("sleeve_row")
        self.gridLayout.addWidget(self.sleeve_row, 7, 0, 1, 1)
        self.hem_size = QtWidgets.QLabel(self.table)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.hem_size.sizePolicy().hasHeightForWidth())
        self.hem_size.setSizePolicy(sizePolicy)
        self.hem_size.setMinimumSize(QtCore.QSize(200, 45))
        font = QtGui.QFont(JetBrainsMonoMedium)
        font.setPointSize(11)
        font.setWeight(55)
        self.hem_size.setFont(font)
        self.hem_size.setStyleSheet("QLabel {\n" "    border: none;\n" "    border-top: 1px solid black;\n" "}")
        self.hem_size.setAlignment(QtCore.Qt.AlignCenter)
        self.hem_size.setObjectName("hem_size")
        self.gridLayout.addWidget(self.hem_size, 3, 1, 1, 1)
        self.name_column = QtWidgets.QLabel(self.table)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.name_column.sizePolicy().hasHeightForWidth())
        self.name_column.setSizePolicy(sizePolicy)
        self.name_column.setMinimumSize(QtCore.QSize(200, 45))
        font = QtGui.QFont()
        font = QtGui.QFont(JetBrainsMonoBold)
        font.setPointSize(13)
        font.setBold(False)
        font.setWeight(75)
        self.name_column.setFont(font)
        self.name_column.setStyleSheet(
            "QLabel {\n" "    border: none;\n" "    color: white;\n" "    background-color: #a388ee;\n" "}"
        )
        self.name_column.setAlignment(QtCore.Qt.AlignCenter)
        self.name_column.setObjectName("name_column")
        self.gridLayout.addWidget(self.name_column, 0, 0, 1, 1)
        self.size_column = QtWidgets.QLabel(self.table)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.size_column.sizePolicy().hasHeightForWidth())
        self.size_column.setSizePolicy(sizePolicy)
        self.size_column.setMinimumSize(QtCore.QSize(200, 45))
        font = QtGui.QFont(JetBrainsMonoBold)
        font.setPointSize(13)
        font.setBold(False)
        font.setWeight(75)
        self.size_column.setFont(font)
        self.size_column.setStyleSheet(
            "QLabel {\n" "    border: none;\n" "    color: white;\n" "    background-color: #a388ee;\n" "}"
        )
        self.size_column.setAlignment(QtCore.Qt.AlignCenter)
        self.size_column.setObjectName("size_column")
        self.gridLayout.addWidget(self.size_column, 0, 1, 1, 1)
        self.shoulder_row = QtWidgets.QLabel(self.table)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.shoulder_row.sizePolicy().hasHeightForWidth())
        self.shoulder_row.setSizePolicy(sizePolicy)
        self.shoulder_row.setMinimumSize(QtCore.QSize(200, 45))
        font = QtGui.QFont(JetBrainsMonoMedium)
        font.setPointSize(11)
        font.setWeight(55)
        self.shoulder_row.setFont(font)
        self.shoulder_row.setStyleSheet("QLabel {\n" "    border: none;\n" "    border-top: 1px solid black;\n" "}")
        self.shoulder_row.setAlignment(QtCore.Qt.AlignCenter)
        self.shoulder_row.setObjectName("shoulder_row")
        self.gridLayout.addWidget(self.shoulder_row, 4, 0, 1, 1)
        self.verticalLayout_3.addWidget(self.table)
        spacerItem6 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_3.addItem(spacerItem6)
        self.bcontainer = QtWidgets.QWidget(self.rcontainer)
        self.bcontainer.setMinimumSize(QtCore.QSize(0, 0))
        self.bcontainer.setObjectName("bcontainer")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.bcontainer)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setSpacing(0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.start_btn = QtWidgets.QPushButton(self.bcontainer)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.start_btn.sizePolicy().hasHeightForWidth())
        self.start_btn.setSizePolicy(sizePolicy)
        self.start_btn.setMinimumSize(QtCore.QSize(175, 40))
        self.start_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        font = QtGui.QFont(JetBrainsMonoBold)
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.start_btn.setFont(font)
        self.start_btn.setStyleSheet(
            "QPushButton{\n"
            "    color: black;\n"
            "    background-color: #f3f5fc;\n"
            "    padding: 8px;\n"
            "    border: 2px solid black;\n"
            "}\n"
            "\n"
            "QPushButton:hover{\n"
            "    background-color: #c3c5cc; \n"
            "}\n"
            "\n"
            "QPushButton:pressed{\n"
            "    background-color: #a388ee;\n"
            "}\n"
            "QPushButton:disabled{\n"
            "   color: #959595;\n"
            "background-color: #dbdcd7;"
            "}"
        )
        self.start_btn.setFlat(True)
        self.start_btn.setObjectName("start_btn")
        self.horizontalLayout_4.addWidget(self.start_btn)
        spacerItem7 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Minimum
        )
        self.horizontalLayout_4.addItem(spacerItem7)
        self.clear_btn = QtWidgets.QPushButton(self.bcontainer)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.clear_btn.sizePolicy().hasHeightForWidth())
        self.clear_btn.setSizePolicy(sizePolicy)
        self.clear_btn.setMinimumSize(QtCore.QSize(175, 40))
        self.clear_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        font = QtGui.QFont(JetBrainsMonoBold)
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.clear_btn.setFont(font)
        self.clear_btn.setStyleSheet(
            "QPushButton{\n"
            "    color: black;\n"
            "    background-color: #f3f5fc;\n"
            "    padding: 8px;\n"
            "    border: 2px solid black;\n"
            "}\n"
            "\n"
            "QPushButton:hover{\n"
            "    background-color: #c3c5cc; \n"
            "}\n"
            "\n"
            "QPushButton:pressed{\n"
            "    background-color: #ff6b6b;\n"
            "}\n"
            "QPushButton:disabled{\n"
            "   color: #959595;\n"
            "background-color: #dbdcd7;"
            "}"
        )
        self.clear_btn.setFlat(True)
        self.clear_btn.setObjectName("clear_btn")
        self.horizontalLayout_4.addWidget(self.clear_btn)
        self.verticalLayout_3.addWidget(self.bcontainer)
        spacerItem8 = QtWidgets.QSpacerItem(20, 39, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_3.addItem(spacerItem8)
        self.horizontalLayout_3.addWidget(self.rcontainer)
        spacerItem7 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Minimum
        )
        self.horizontalLayout_3.addItem(spacerItem7)
        self.horizontalLayout_2.addWidget(self.container)
        self.verticalLayout_2.addWidget(self.tcontainer)
        spacerItem8 = QtWidgets.QSpacerItem(
            20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.MinimumExpanding
        )
        self.verticalLayout_2.addItem(spacerItem8)
        self.verticalLayout.addWidget(self.body)
        self.setCentralWidget(self.centralwidget)

        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self)

    def loadFont(self, fontPath):
        fontDB = QtGui.QFontDatabase()
        fontID = fontDB.addApplicationFont(fontPath)
        if fontID == -1:
            return None
        fontFamilies = fontDB.applicationFontFamilies(fontID)
        if fontFamilies is None:
            return None
        return fontFamilies[0]

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("MainWindow", "TeeSize"))
        self.logo.setText(_translate("MainWindow", "TEESIZE"))

        self.name_column.setText(_translate("MainWindow", "Name"))
        self.size_column.setText(_translate("MainWindow", "Size"))

        self.chest_row.setText(_translate("MainWindow", "Chest"))
        self.waist_row.setText(_translate("MainWindow", "Waist"))
        self.hem_row.setText(_translate("MainWindow", "Hem"))
        self.shoulder_row.setText(_translate("MainWindow", "Shoulder"))
        self.length_row.setText(_translate("MainWindow", "Length"))
        self.neck_row.setText(_translate("MainWindow", "Neck"))
        self.sleeve_row.setText(_translate("MainWindow", "Sleeve"))
        self.cuff_row.setText(_translate("MainWindow", "Cuff"))

        self.chest_size.setText(_translate("MainWindow", "—"))
        self.waist_size.setText(_translate("MainWindow", "—"))
        self.hem_size.setText(_translate("MainWindow", "—"))
        self.shoulder_size.setText(_translate("MainWindow", "—"))
        self.length_size.setText(_translate("MainWindow", "—"))
        self.neck_size.setText(_translate("MainWindow", "—"))
        self.sleeve_size.setText(_translate("MainWindow", "—"))
        self.cuff_size.setText(_translate("MainWindow", "—"))

        self.start_btn.setText(_translate("MainWindow", "Start"))
        self.clear_btn.setText(_translate("MainWindow", "Clear"))