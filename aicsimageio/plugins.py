

import sys
if sys.version_info < (3, 10):
    from importlib_metadata import entry_points, version, distribution
else:
    from importlib.metadata import entry_points, version, distribution

# from setuptools import pkg_resources
# key: "entrypoint", "metadata"
plugin_cache = []


def get_plugins():
    # if we have hardcoded plugins we could add them here first
    # get_provider(package_or_requirement)
    plugins = entry_points(group='aicsimageio.readers')
    for plugin in plugins:
        # ReaderMetadata knows how to instantiate the actual Reader
        plugin_cache.append({"readermeta":plugin.load().ReaderMetadata, "entrypoint": plugin})
    return plugin_cache


def dump_plugins():
    get_plugins()
    for plugin in plugin_cache:
        ep = plugin["entrypoint"]
        print(ep.name)
        print(f"  Author  : {ep.dist.metadata['author']}")
        print(f"  Version : {ep.dist.version}")
        print(f"  License : {ep.dist.metadata['license']}")
        # print(f"  Description : {ep.dist.metadata['description']}")
        reader_meta = plugin["readermeta"]
        exts = ", ".join(reader_meta.get_supported_extensions())
        print(f"  Supported Extensions : {exts}")
