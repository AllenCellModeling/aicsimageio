
from typing import Dict, List, Optional, NamedTuple

import sys
if sys.version_info < (3, 10):
    from importlib_metadata import entry_points, version, distribution, EntryPoint
else:
    from importlib.metadata import entry_points, version, distribution, EntryPoint

from base_image_reader import ReaderMetadata


class PluginEntry(NamedTuple):
    entrypoint: EntryPoint
    metadata: ReaderMetadata


# global cache of plugins
plugin_cache: List[PluginEntry] = []
# global cache of plugins by extension
# note there can be multiple readers for the same extension
plugins_by_ext: Dict[str, List[ReaderMetadata]] = {}


def get_plugins():
    plugins = entry_points(group='aicsimageio.readers')
    for plugin in plugins:
        # ReaderMetadata knows how to instantiate the actual Reader
        reader_meta = plugin.load().ReaderMetadata
        plugin_cache.append({
            "readermeta": reader_meta,
            "entrypoint": plugin
        })
        exts = reader_meta.get_supported_extensions()
        for ext in exts:
            if ext not in plugins_by_ext:
                plugins_by_ext[ext] = []
            # TODO insert in sorted order (sorted by most recently installed)
            plugins_by_ext[ext].append(reader_meta)

    return plugin_cache


def dump_plugins():
    # TODO don't call get_plugins every time
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
    print("Plugins for extensions:")
    sorted_exts = sorted(plugins_by_ext.keys())
    for ext in sorted_exts:
        plugins = plugins_by_ext[ext]
        print(f"{ext}: {plugins}")


def find_reader_for_path(path: str) -> Optional[ReaderMetadata]:
    # this is not great and there is probably a cleverer way to do this

    # try to match on the longest possible registered extension
    exts = sorted(plugins_by_ext.keys(), key=len, reverse=True)
    for ext in exts:
        if path.endswith(ext):
            candidates = plugins_by_ext[ext]
            # TODO sort by installed date
            return candidates[0]
    # didn't find a reader for this extension
    return None
