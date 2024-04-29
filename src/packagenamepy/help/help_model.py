"""single help module"""

import webbrowser
from packagenamepy.configuration import get_data


def help_function(context):
    """
    open a browser with the appropriate help page
    """
    help_url = get_data("global.other", "help_url")
    if context:
        webbrowser.open(help_url)
