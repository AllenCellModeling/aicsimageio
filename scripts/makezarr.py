from distributed import LocalCluster, Client
import os
import s3fs

from aicsimageio.writers import OmeZarrWriter
from aicsimageio import AICSImage
from aicsimageio.dimensions import DimensionNames


def write_scene(storeroot, scenename, sceneindex, img):
    print(scenename)
    print(storeroot)
    img.set_scene(sceneindex)
    pps = img.physical_pixel_sizes
    cn = img.channel_names

    data = img.get_image_dask_data("TCZYX")
    print(data.shape)
    writer = OmeZarrWriter(storeroot)

    # construct some per-channel lists to feed in to the writer.
    # hardcoding to 9 for now
    channel_colors = [0xff0000, 0x00ff00, 0x0000ff, 0xffff00, 0xff00ff, 0x00ffff, 0x880000, 0x008800, 0x000088]

    writer.write_image(
        image_data=data,  # : types.ArrayLike,  # must be 5D TCZYX
        image_name=scenename,  #: str,
        physical_pixel_sizes=pps,  # : Optional[types.PhysicalPixelSizes],
        channel_names=cn,  # : Optional[List[str]],
        channel_colors=channel_colors,  # : Optional[List[int]],
        scale_num_levels=4,  # : int = 1,
        scale_factor=2.0  # : float = 2.0,
    )    


def do_main():
    chunk_dims = [
        DimensionNames.SpatialY,
        DimensionNames.SpatialX,
        DimensionNames.Samples,
    ]
    # filepath = "//allen/aics/assay-dev/MicroscopyData/Frick/2022/20220217/LLS/AD00000198/Lamin_multi-06-Deskew-28.czi"

    # filepath = "//allen/programs/allencell/data/proj0/935/c72/962/3cc/7a4/37b/f4f/6d8/69c/91a/71/20200323_F01_001.czi"
    # filename = "20200323_F01_001"

    # filepath = "E:\\data\\AICS-11_16415.ome.tif"
    filepath = "//allen/aics/animated-cell/Allen-Cell-Explorer/Allen-Cell-Explorer_1.4.0/Cell-Viewer_Data/AICS-11/AICS-11_16415.ome.tif"
    filename = "FOV_16415"

    img = AICSImage(filepath, chunk_dims=chunk_dims)

    os.environ["AWS_PROFILE"] = "animatedcell"
    os.environ["AWS_DEFAULT_REGION"] = "us-west-2"
    # cluster = LocalCluster(processes=True)
    cluster = LocalCluster(n_workers=4, processes=True, threads_per_worker=1)
    # cluster = LocalCluster(memory_limit="7GB")  # threaded instead of multiprocess
    # cluster = LocalCluster(n_workers=4, processes=True, threads_per_worker=1, memory_limit="12GB")
    client = Client(cluster)

    # print some data about the image we loaded
    scenes = img.scenes
    print(scenes)
    print(str(len(scenes)))

    print(img.channel_names)
    print(img.physical_pixel_sizes)

    # TODO: add generated thumbnail image with name "thumbnail" as 2d rgb data? or 3 channels?
    # TODO compression?

    s3 = s3fs.S3FileSystem(anon=False, config_kwargs={"connect_timeout": 60})

    # known scenes we want to write:
    # babybear, goldilocks, mamabear, papabear
    # scene_indices = [9-1,6-1,5-1,8-1]
    # scene_indices = [9-1]
    # scene_indices = [6-1,5-1,8-1]
    scene_indices = range(len(img.scenes))

    for i in scene_indices:
        scenename = img.scenes[i]
        scenename = scenename.replace(":","_")
        write_scene("s3://animatedcell-test-data/" + filename + "/" + scenename + ".zarr/", scenename, i, img)


if __name__ == "__main__":
    do_main()
