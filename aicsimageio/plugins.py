from datetime import datetime
import os
import sys

if sys.version_info < (3, 10):
    from importlib_metadata import entry_points, EntryPoint
else:
    from importlib.metadata import entry_points, EntryPoint
from typing import Dict, List, Optional, NamedTuple

from base_image_reader import ReaderMetadata
from base_image_reader.reader import Reader


class PluginEntry(NamedTuple):
    entrypoint: EntryPoint
    metadata: ReaderMetadata
    timestamp: datetime


# global cache of plugins
plugin_cache: List[PluginEntry] = []
# global cache of plugins by extension
# note there can be multiple readers for the same extension
plugins_by_ext: Dict[str, List[PluginEntry]] = {}


def insert_sorted_by_timestamp(list: List[PluginEntry], item: PluginEntry):
    for i, other in enumerate(list):
        if item.timestamp > other.timestamp:
            list.insert(i, item)
            return
    list.append(item)


def add_plugin(pluginentry: PluginEntry):
    plugin_cache.append(pluginentry)
    exts = pluginentry.metadata.get_supported_extensions()
    for ext in exts:
        if ext not in plugins_by_ext:
            plugins_by_ext[ext] = [pluginentry]
            continue

        # insert in sorted order (sorted by most recently installed)
        pluginlist = plugins_by_ext[ext]
        insert_sorted_by_timestamp(pluginlist, pluginentry)


def get_plugins():
    plugins = entry_points(group="aicsimageio.readers")
    for plugin in plugins:
        # ReaderMetadata knows how to instantiate the actual Reader
        reader_meta = plugin.load().ReaderMetadata
        if plugin.dist.files is not None:
            firstfile = plugin.dist.files[0]
            timestamp = os.path.getmtime(firstfile.locate().parent)
        else:
            timestamp = 0
        pluginentry = PluginEntry(plugin, reader_meta, timestamp)
        add_plugin(pluginentry)

    return plugin_cache


def dump_plugins():
    # TODO don't call get_plugins every time
    get_plugins()
    for plugin in plugin_cache:
        ep = plugin.entrypoint
        print(ep.name)
        print(f"  Author  : {ep.dist.metadata['author']}")
        print(f"  Version : {ep.dist.version}")
        print(f"  License : {ep.dist.metadata['license']}")
        firstfile = ep.dist.files[0]
        t = datetime.fromtimestamp(os.path.getmtime(firstfile.locate().parent))
        print(f"  Date    : {t}")
        # print(f"  Description : {ep.dist.metadata['description']}")
        reader_meta = plugin.metadata
        exts = ", ".join(reader_meta.get_supported_extensions())
        print(f"  Supported Extensions : {exts}")
    print("Plugins for extensions:")
    sorted_exts = sorted(plugins_by_ext.keys())
    for ext in sorted_exts:
        plugins = plugins_by_ext[ext]
        print(f"{ext}: {plugins}")


def find_reader_for_path(path: str) -> Optional[Reader]:
    candidates = find_readers_for_path(path)
    for candidate in candidates:
        reader = candidate.metadata.get_reader()
        if reader.is_supported_image(
            path,
            # TODO fs_kwargs=fs_kwargs,
        ):
            return reader
    return None

    # try to match on the longest possible registered extension
    # exts = sorted(plugins_by_ext.keys(), key=len, reverse=True)
    # for ext in exts:
    #     if path.endswith(ext):
    #         candidates = plugins_by_ext[ext]
    #         # TODO select a candidate by some criteria?
    #         return candidates[0]
    # # didn't find a reader for this extension
    # return None


def find_readers_for_path(path: str) -> List[PluginEntry]:
    candidates = []
    # try to match on the longest possible registered extension first
    exts = sorted(plugins_by_ext.keys(), key=len, reverse=True)
    for ext in exts:
        if path.endswith(ext):
            candidates = candidates + plugins_by_ext[ext]
    print(candidates)
    return candidates
