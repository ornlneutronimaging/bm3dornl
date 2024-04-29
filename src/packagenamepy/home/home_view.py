"""PyQt widget for the main tab"""

from qtpy.QtWidgets import QWidget, QHBoxLayout


class Home(QWidget):  # pylint: disable=too-many-public-methods
    """Main widget"""

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QHBoxLayout()
        self.setLayout(layout)
