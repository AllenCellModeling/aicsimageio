

import sys
if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points


plugin_cache = {}


def get_plugins():
    # if we have hardcoded plugins we could add them here first

    discovered_plugins = entry_points(group='aicsimageio.readers')
    # print(discovered_plugins)
    # print(dir(discovered_plugins))
    # print(type(discovered_plugins))
    for plugin in discovered_plugins:
        plugin_cache[plugin.name] = plugin.load().Reader
        # plugin.load.Reader is the reader class
        # plugin.load.ReaderInfo is the reader metadata class
    # print(plugin_cache)
    return discovered_plugins, plugin_cache
