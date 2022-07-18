

import sys
if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points


plugin_cache = {}


def get_plugins():
    discovered_plugins = entry_points(group='aicsimageio.readers')
    # print(discovered_plugins)
    # print(dir(discovered_plugins))
    # print(type(discovered_plugins))
    for plugin in discovered_plugins:
        plugin_cache[plugin.name] = plugin.load()
    # print(plugin_cache)
    return discovered_plugins, plugin_cache
