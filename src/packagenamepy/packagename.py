"""
Main Qt application
"""

import sys
from qtpy.QtWidgets import QApplication, QMainWindow

from mantid.kernel import Logger
from mantidqt.gui_helper import set_matplotlib_backend

# make sure matplotlib is correctly set before we import shiver
set_matplotlib_backend()

# make sure the algorithms have been loaded so they are available to the AlgorithmManager
import mantid.simpleapi  # noqa: F401, E402 pylint: disable=unused-import, wrong-import-position

from packagenamepy.configuration import Configuration  # noqa: E402 pylint: disable=wrong-import-position
from packagenamepy.version import __version__  # noqa: E402 pylint: disable=wrong-import-position
from packagenamepy.mainwindow import MainWindow  # noqa: E402 pylint: disable=wrong-import-position

logger = Logger("PACKAGENAME")


class PackageName(QMainWindow):
    """Main Package window"""

    __instance = None

    def __new__(cls):
        if PackageName.__instance is None:
            PackageName.__instance = QMainWindow.__new__(cls)  # pylint: disable=no-value-for-parameter
        return PackageName.__instance

    def __init__(self, parent=None):
        super().__init__(parent)
        logger.information(f"PackageName version: {__version__}")
        config = Configuration()

        if not config.is_valid():
            msg = (
                "Error with configuration settings!",
                f"Check and update your file: {config.config_file_path}",
                "with the latest settings found here:",
                f"{config.template_file_path} and start the application again.",
            )

            print(" ".join(msg))
            sys.exit(-1)
        self.setWindowTitle(f"PACKAGENAME - {__version__}")
        self.main_window = MainWindow(self)
        self.setCentralWidget(self.main_window)


def gui():
    """
    Main entry point for Qt application
    """
    input_flags = sys.argv[1::]
    if "--v" in input_flags or "--version" in input_flags:
        print(__version__)
        sys.exit()
    else:
        app = QApplication(sys.argv)
        window = PackageName()
        window.show()
        sys.exit(app.exec_())
