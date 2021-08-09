Search.setIndex({docnames:["aicsimageio","aicsimageio.readers","aicsimageio.vendor","aicsimageio.writers","benchmarks","changelog","contributing","index","installation","modules"],envversion:{"sphinx.domains.c":1,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":1,"sphinx.domains.index":1,"sphinx.domains.javascript":1,"sphinx.domains.math":2,"sphinx.domains.python":1,"sphinx.domains.rst":1,"sphinx.domains.std":1,"sphinx.ext.viewcode":1,sphinx:56},filenames:["aicsimageio.rst","aicsimageio.readers.rst","aicsimageio.vendor.rst","aicsimageio.writers.rst","benchmarks.md","changelog.rst","contributing.rst","index.rst","installation.rst","modules.rst"],objects:{"":{aicsimageio:[0,0,0,"-"]},"aicsimageio.aics_image":{AICSImage:[0,1,1,""],imread:[0,3,1,""],imread_dask:[0,3,1,""]},"aicsimageio.aics_image.AICSImage":{client:[0,2,1,""],close:[0,2,1,""],cluster:[0,2,1,""],dask_data:[0,2,1,""],data:[0,2,1,""],determine_reader:[0,2,1,""],get_channel_names:[0,2,1,""],get_image_dask_data:[0,2,1,""],get_image_data:[0,2,1,""],get_physical_pixel_size:[0,2,1,""],metadata:[0,2,1,""],reader:[0,2,1,""],shape:[0,2,1,""],size:[0,2,1,""],size_c:[0,2,1,""],size_s:[0,2,1,""],size_t:[0,2,1,""],size_x:[0,2,1,""],size_y:[0,2,1,""],size_z:[0,2,1,""],view_napari:[0,2,1,""]},"aicsimageio.buffer_reader":{BufferReader:[0,1,1,""]},"aicsimageio.buffer_reader.BufferReader":{INTEL_ENDIAN:[0,4,1,""],MOTOROLA_ENDIAN:[0,4,1,""],read_bytes:[0,2,1,""],read_uint16:[0,2,1,""],read_uint32:[0,2,1,""],read_uint64:[0,2,1,""],reset:[0,2,1,""]},"aicsimageio.constants":{Dimensions:[0,1,1,""]},"aicsimageio.constants.Dimensions":{Channel:[0,4,1,""],DefaultOrder:[0,4,1,""],DefaultOrderList:[0,4,1,""],Scene:[0,4,1,""],SpatialX:[0,4,1,""],SpatialY:[0,4,1,""],SpatialZ:[0,4,1,""],Time:[0,4,1,""]},"aicsimageio.dask_utils":{cluster_and_client:[0,3,1,""],shutdown_cluster_and_client:[0,3,1,""],spawn_cluster_and_client:[0,3,1,""]},"aicsimageio.exceptions":{ConflictingArgumentsError:[0,5,1,""],InconsistentPixelType:[0,5,1,""],InconsistentShapeError:[0,5,1,""],InvalidDimensionOrderingError:[0,5,1,""],UnsupportedFileFormatError:[0,5,1,""]},"aicsimageio.readers":{arraylike_reader:[1,0,0,"-"],czi_reader:[1,0,0,"-"],default_reader:[1,0,0,"-"],lif_reader:[1,0,0,"-"],ome_tiff_reader:[1,0,0,"-"],reader:[1,0,0,"-"],tiff_reader:[1,0,0,"-"]},"aicsimageio.readers.arraylike_reader":{ArrayLikeReader:[1,1,1,""]},"aicsimageio.readers.arraylike_reader.ArrayLikeReader":{dims:[1,2,1,""],metadata:[1,2,1,""]},"aicsimageio.readers.czi_reader":{CziReader:[1,1,1,""]},"aicsimageio.readers.czi_reader.CziReader":{ZEISS_10BYTE:[1,4,1,""],ZEISS_2BYTE:[1,4,1,""],dims:[1,2,1,""],dtype:[1,2,1,""],get_channel_names:[1,2,1,""],get_physical_pixel_size:[1,2,1,""],metadata:[1,2,1,""],size_c:[1,2,1,""],size_s:[1,2,1,""],size_t:[1,2,1,""],size_x:[1,2,1,""],size_y:[1,2,1,""],size_z:[1,2,1,""]},"aicsimageio.readers.default_reader":{DefaultReader:[1,1,1,""]},"aicsimageio.readers.default_reader.DefaultReader":{dims:[1,2,1,""],get_channel_names:[1,2,1,""],metadata:[1,2,1,""]},"aicsimageio.readers.lif_reader":{LifReader:[1,1,1,""]},"aicsimageio.readers.lif_reader.LifReader":{LIF_MAGIC_BYTE:[1,4,1,""],LIF_MEMORY_BYTE:[1,4,1,""],dims:[1,2,1,""],dtype:[1,2,1,""],get_channel_names:[1,2,1,""],get_physical_pixel_size:[1,2,1,""],get_pixel_type:[1,2,1,""],metadata:[1,2,1,""],size_c:[1,2,1,""],size_s:[1,2,1,""],size_t:[1,2,1,""],size_x:[1,2,1,""],size_y:[1,2,1,""],size_z:[1,2,1,""]},"aicsimageio.readers.ome_tiff_reader":{OmeTiffReader:[1,1,1,""]},"aicsimageio.readers.ome_tiff_reader.OmeTiffReader":{get_channel_names:[1,2,1,""],get_physical_pixel_size:[1,2,1,""],is_ome:[1,2,1,""],metadata:[1,2,1,""],size_c:[1,2,1,""],size_s:[1,2,1,""],size_t:[1,2,1,""],size_x:[1,2,1,""],size_y:[1,2,1,""],size_z:[1,2,1,""]},"aicsimageio.readers.reader":{Reader:[1,1,1,""],use_dask:[1,3,1,""]},"aicsimageio.readers.reader.Reader":{client:[1,2,1,""],close:[1,2,1,""],cluster:[1,2,1,""],dask_data:[1,2,1,""],data:[1,2,1,""],dims:[1,2,1,""],get_channel_names:[1,2,1,""],get_image_dask_data:[1,2,1,""],get_image_data:[1,2,1,""],get_physical_pixel_size:[1,2,1,""],guess_dim_order:[1,2,1,""],is_this_type:[1,2,1,""],metadata:[1,2,1,""],shape:[1,2,1,""],size:[1,2,1,""]},"aicsimageio.readers.tiff_reader":{TiffReader:[1,1,1,""]},"aicsimageio.readers.tiff_reader.TiffReader":{dims:[1,2,1,""],dtype:[1,2,1,""],get_image_description:[1,2,1,""],load_slice:[1,2,1,""],metadata:[1,2,1,""]},"aicsimageio.transforms":{reshape_data:[0,3,1,""],transpose_to_dims:[0,3,1,""]},"aicsimageio.types":{LoadResults:[0,1,1,""]},"aicsimageio.types.LoadResults":{data:[0,4,1,""],dims:[0,4,1,""],metadata:[0,4,1,""]},"aicsimageio.vendor":{omexml:[2,0,0,"-"]},"aicsimageio.vendor.omexml":{DO_XYTZC:[2,6,1,""],MPI_CMYK:[2,6,1,""],OMEXML:[2,1,1,""],OM_ARTIST:[2,6,1,""],OM_BITS_PER_SAMPLE:[2,6,1,""],OM_CELL_LENGTH:[2,6,1,""],OM_CELL_WIDTH:[2,6,1,""],OM_DATE_TIME:[2,6,1,""],OM_DOCUMENT_NAME:[2,6,1,""],OM_FILL_ORDER:[2,6,1,""],OM_FREE_BYTECOUNTS:[2,6,1,""],OM_FREE_OFFSETS:[2,6,1,""],OM_GRAY_RESPONSE_CURVE:[2,6,1,""],OM_GRAY_RESPONSE_UNIT:[2,6,1,""],OM_HOST_COMPUTER:[2,6,1,""],OM_IMAGE_LENGTH:[2,6,1,""],OM_IMAGE_WIDTH:[2,6,1,""],OM_INK_SET:[2,6,1,""],OM_MAKE:[2,6,1,""],OM_MAX_SAMPLE_VALUE:[2,6,1,""],OM_MIN_SAMPLE_VALUE:[2,6,1,""],OM_MODEL:[2,6,1,""],OM_NEW_SUBFILE_TYPE:[2,6,1,""],OM_ORIENTATION:[2,6,1,""],OM_PAGE_NUMBER:[2,6,1,""],OM_PREDICTOR:[2,6,1,""],OM_RESOLUTION_UNIT:[2,6,1,""],OM_SAMPLES_PER_PIXEL:[2,6,1,""],OM_SOFTWARE:[2,6,1,""],OM_T4_OPTIONS:[2,6,1,""],OM_T6_OPTIONS:[2,6,1,""],OM_THRESHHOLDING:[2,6,1,""],OM_TILE_BYTE_COUNT:[2,6,1,""],OM_TILE_LENGTH:[2,6,1,""],OM_TILE_OFFSETS:[2,6,1,""],OM_TILE_WIDTH:[2,6,1,""],OM_TRANSFER_FUNCTION:[2,6,1,""],OM_WHITE_POINT:[2,6,1,""],OM_X_POSITION:[2,6,1,""],OM_X_RESOLUTION:[2,6,1,""],OM_Y_POSITION:[2,6,1,""],OM_Y_RESOLUTION:[2,6,1,""],PC_PLANAR:[2,6,1,""],PI_CFA_ARRAY:[2,6,1,""],get_float_attr:[2,3,1,""],get_int_attr:[2,3,1,""],get_namespaces:[2,3,1,""],get_pixel_type:[2,3,1,""],get_text:[2,3,1,""],make_text_node:[2,3,1,""],page_name_original_metadata:[2,3,1,""],qn:[2,3,1,""],set_text:[2,3,1,""],split_qn:[2,3,1,""],xsd_now:[2,3,1,""]},"aicsimageio.vendor.omexml.OMEXML":{Channel:[2,1,1,""],Image:[2,1,1,""],OriginalMetadata:[2,1,1,""],Pixels:[2,1,1,""],Plane:[2,1,1,""],Plate:[2,1,1,""],PlatesDucktype:[2,1,1,""],StructuredAnnotations:[2,1,1,""],TiffData:[2,1,1,""],Well:[2,1,1,""],WellSample:[2,1,1,""],WellSampleDucktype:[2,1,1,""],WellsDucktype:[2,1,1,""],get_image_count:[2,2,1,""],get_ns:[2,2,1,""],image:[2,2,1,""],image_count:[2,2,1,""],plates:[2,2,1,""],root_node:[2,2,1,""],set_image_count:[2,2,1,""],structured_annotations:[2,2,1,""],to_xml:[2,2,1,""]},"aicsimageio.vendor.omexml.OMEXML.Channel":{Color:[2,2,1,""],ID:[2,2,1,""],Name:[2,2,1,""],SamplesPerPixel:[2,2,1,""],get_Color:[2,2,1,""],get_ID:[2,2,1,""],get_Name:[2,2,1,""],get_SamplesPerPixel:[2,2,1,""],set_Color:[2,2,1,""],set_ID:[2,2,1,""],set_Name:[2,2,1,""],set_SamplesPerPixel:[2,2,1,""]},"aicsimageio.vendor.omexml.OMEXML.Image":{AcquisitionDate:[2,2,1,""],ID:[2,2,1,""],Name:[2,2,1,""],Pixels:[2,2,1,""],get_AcquisitionDate:[2,2,1,""],get_ID:[2,2,1,""],get_Name:[2,2,1,""],set_AcquisitionDate:[2,2,1,""],set_ID:[2,2,1,""],set_Name:[2,2,1,""]},"aicsimageio.vendor.omexml.OMEXML.OriginalMetadata":{has_key:[2,2,1,""],iteritems:[2,2,1,""],keys:[2,2,1,""]},"aicsimageio.vendor.omexml.OMEXML.Pixels":{Channel:[2,2,1,""],DimensionOrder:[2,2,1,""],ID:[2,2,1,""],PhysicalSizeX:[2,2,1,""],PhysicalSizeY:[2,2,1,""],PhysicalSizeZ:[2,2,1,""],PixelType:[2,2,1,""],Plane:[2,2,1,""],SizeC:[2,2,1,""],SizeT:[2,2,1,""],SizeX:[2,2,1,""],SizeY:[2,2,1,""],SizeZ:[2,2,1,""],TiffData:[2,2,1,""],append_channel:[2,2,1,""],channel_count:[2,2,1,""],get_DimensionOrder:[2,2,1,""],get_ID:[2,2,1,""],get_PhysicalSizeX:[2,2,1,""],get_PhysicalSizeY:[2,2,1,""],get_PhysicalSizeZ:[2,2,1,""],get_PixelType:[2,2,1,""],get_SizeC:[2,2,1,""],get_SizeT:[2,2,1,""],get_SizeX:[2,2,1,""],get_SizeY:[2,2,1,""],get_SizeZ:[2,2,1,""],get_channel_count:[2,2,1,""],get_channel_names:[2,2,1,""],get_plane_count:[2,2,1,""],get_planes_of_channel:[2,2,1,""],plane_count:[2,2,1,""],populate_TiffData:[2,2,1,""],remove_channel:[2,2,1,""],set_DimensionOrder:[2,2,1,""],set_ID:[2,2,1,""],set_PhysicalSizeX:[2,2,1,""],set_PhysicalSizeY:[2,2,1,""],set_PhysicalSizeZ:[2,2,1,""],set_PixelType:[2,2,1,""],set_SizeC:[2,2,1,""],set_SizeT:[2,2,1,""],set_SizeX:[2,2,1,""],set_SizeY:[2,2,1,""],set_SizeZ:[2,2,1,""],set_channel_count:[2,2,1,""],set_plane_count:[2,2,1,""]},"aicsimageio.vendor.omexml.OMEXML.Plane":{DeltaT:[2,2,1,""],ExposureTime:[2,2,1,""],PositionX:[2,2,1,""],PositionY:[2,2,1,""],PositionZ:[2,2,1,""],TheC:[2,2,1,""],TheT:[2,2,1,""],TheZ:[2,2,1,""],get_DeltaT:[2,2,1,""],get_PositionX:[2,2,1,""],get_PositionY:[2,2,1,""],get_PositionZ:[2,2,1,""],get_TheC:[2,2,1,""],get_TheT:[2,2,1,""],get_TheZ:[2,2,1,""],set_DeltaT:[2,2,1,""],set_PositionX:[2,2,1,""],set_PositionY:[2,2,1,""],set_PositionZ:[2,2,1,""],set_TheC:[2,2,1,""],set_TheT:[2,2,1,""],set_TheZ:[2,2,1,""]},"aicsimageio.vendor.omexml.OMEXML.Plate":{ColumnNamingConvention:[2,2,1,""],Columns:[2,2,1,""],Description:[2,2,1,""],ExternalIdentifier:[2,2,1,""],ID:[2,2,1,""],Name:[2,2,1,""],RowNamingConvention:[2,2,1,""],Rows:[2,2,1,""],Status:[2,2,1,""],Well:[2,2,1,""],WellOriginX:[2,2,1,""],WellOriginY:[2,2,1,""],get_ColumnNamingConvention:[2,2,1,""],get_Columns:[2,2,1,""],get_Description:[2,2,1,""],get_ExternalIdentifier:[2,2,1,""],get_ID:[2,2,1,""],get_Name:[2,2,1,""],get_RowNamingConvention:[2,2,1,""],get_Rows:[2,2,1,""],get_Status:[2,2,1,""],get_Well:[2,2,1,""],get_WellOriginX:[2,2,1,""],get_WellOriginY:[2,2,1,""],get_well_name:[2,2,1,""],set_ColumnNamingConvention:[2,2,1,""],set_Columns:[2,2,1,""],set_Description:[2,2,1,""],set_ExternalIdentifier:[2,2,1,""],set_ID:[2,2,1,""],set_Name:[2,2,1,""],set_RowNamingConvention:[2,2,1,""],set_Rows:[2,2,1,""],set_Status:[2,2,1,""],set_WellOriginX:[2,2,1,""],set_WellOriginY:[2,2,1,""]},"aicsimageio.vendor.omexml.OMEXML.PlatesDucktype":{newPlate:[2,2,1,""]},"aicsimageio.vendor.omexml.OMEXML.StructuredAnnotations":{OriginalMetadata:[2,2,1,""],add_original_metadata:[2,2,1,""],get_original_metadata_refs:[2,2,1,""],get_original_metadata_value:[2,2,1,""],has_key:[2,2,1,""],has_original_metadata:[2,2,1,""],iter_original_metadata:[2,2,1,""],keys:[2,2,1,""]},"aicsimageio.vendor.omexml.OMEXML.TiffData":{FirstC:[2,2,1,""],FirstT:[2,2,1,""],FirstZ:[2,2,1,""],IFD:[2,2,1,""],PlaneCount:[2,2,1,""],get_FirstC:[2,2,1,""],get_FirstT:[2,2,1,""],get_FirstZ:[2,2,1,""],get_IFD:[2,2,1,""],get_PlaneCount:[2,2,1,""],set_FirstC:[2,2,1,""],set_FirstT:[2,2,1,""],set_FirstZ:[2,2,1,""],set_IFD:[2,2,1,""],set_PlaneCount:[2,2,1,""]},"aicsimageio.vendor.omexml.OMEXML.Well":{Column:[2,2,1,""],ExternalDescription:[2,2,1,""],ExternalIdentifier:[2,2,1,""],ID:[2,2,1,""],Row:[2,2,1,""],Sample:[2,2,1,""],get_Color:[2,2,1,""],get_Column:[2,2,1,""],get_ExternalDescription:[2,2,1,""],get_ExternalIdentifier:[2,2,1,""],get_ID:[2,2,1,""],get_Row:[2,2,1,""],get_Sample:[2,2,1,""],set_Color:[2,2,1,""],set_Column:[2,2,1,""],set_ExternalDescription:[2,2,1,""],set_ExternalIdentifier:[2,2,1,""],set_ID:[2,2,1,""],set_Row:[2,2,1,""]},"aicsimageio.vendor.omexml.OMEXML.WellSample":{ID:[2,2,1,""],ImageRef:[2,2,1,""],Index:[2,2,1,""],PositionX:[2,2,1,""],PositionY:[2,2,1,""],Timepoint:[2,2,1,""],get_ID:[2,2,1,""],get_ImageRef:[2,2,1,""],get_Index:[2,2,1,""],get_PositionX:[2,2,1,""],get_PositionY:[2,2,1,""],get_Timepoint:[2,2,1,""],set_ID:[2,2,1,""],set_ImageRef:[2,2,1,""],set_Index:[2,2,1,""],set_PositionX:[2,2,1,""],set_PositionY:[2,2,1,""],set_Timepoint:[2,2,1,""]},"aicsimageio.vendor.omexml.OMEXML.WellSampleDucktype":{"new":[2,2,1,""]},"aicsimageio.vendor.omexml.OMEXML.WellsDucktype":{"new":[2,2,1,""]},"aicsimageio.writers":{ome_tiff_writer:[3,0,0,"-"],png_writer:[3,0,0,"-"],writer:[3,0,0,"-"]},"aicsimageio.writers.ome_tiff_writer":{OmeTiffWriter:[3,1,1,""]},"aicsimageio.writers.ome_tiff_writer.OmeTiffWriter":{close:[3,2,1,""],save:[3,2,1,""],save_slice:[3,2,1,""],set_metadata:[3,2,1,""],size_c:[3,2,1,""],size_t:[3,2,1,""],size_x:[3,2,1,""],size_y:[3,2,1,""],size_z:[3,2,1,""]},"aicsimageio.writers.png_writer":{PngWriter:[3,1,1,""]},"aicsimageio.writers.png_writer.PngWriter":{close:[3,2,1,""],save:[3,2,1,""],save_slice:[3,2,1,""]},"aicsimageio.writers.writer":{Writer:[3,1,1,""]},"aicsimageio.writers.writer.Writer":{close:[3,2,1,""],save:[3,2,1,""]},aicsimageio:{aics_image:[0,0,0,"-"],buffer_reader:[0,0,0,"-"],constants:[0,0,0,"-"],dask_utils:[0,0,0,"-"],exceptions:[0,0,0,"-"],readers:[1,0,0,"-"],transforms:[0,0,0,"-"],types:[0,0,0,"-"],vendor:[2,0,0,"-"],writers:[3,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","function","Python function"],"4":["py","attribute","Python attribute"],"5":["py","exception","Python exception"],"6":["py","data","Python data"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:function","4":"py:attribute","5":"py:exception","6":"py:data"},terms:{"116c06f0":2,"11th":0,"14th":2,"1st":0,"287f75db":2,"2nd":0,"3b04334553db":2,"400mb":4,"44d4":2,"44d9":2,"45e0":2,"4acb":2,"4f3d":2,"5e665ed66c1b373a84002227044c7a12a2ecc506b84a730442a5ed798428e26a":4,"5e86c64a0059":2,"65a7cb00a083":2,"7bb9":2,"89ed":2,"8b91":2,"8e63a8a8":2,"97de":2,"abstract":[1,3],"break":5,"byte":[0,1],"case":[1,4],"catch":1,"class":[0,1,2,3,5,7],"default":[0,1,2,3,5,7],"float":[0,1,2],"function":[0,1,3,4,5,7],"h\u00fcrlimann":5,"import":[4,5,7],"int":[0,1,2],"new":[0,2,6],"public":8,"return":[0,1,2,3,5,7],"short":[6,7],"static":[0,1],"switch":5,"throw":3,"true":[1,2,3],"try":4,"while":4,For:[0,2,7],IDs:2,OME:[1,2,3,5,7],Ome:0,The:[0,1,2,3,4,7,8],Then:6,There:[2,4],These:[0,2,4],Use:2,Using:3,Will:3,With:4,a08:2,abc:[1,3],abl:4,about:4,abov:4,accept:0,access:0,accident:5,accompani:3,achiev:4,acquir:2,acquisit:2,acquisitiond:2,across:0,action:[3,5,6],activ:4,actual:2,add:[0,2,5,6],add_original_metadata:2,added:[0,1,5],addit:[5,7],addition:4,address:0,admin:5,after:5,against:4,aic:4,aics_imag:9,aicsimag:[0,4,5,7],aicsimageio:[4,6,8],aicspylibczi:1,alia:0,all:[0,1,4,5,6,7],allencellmodel:[4,7,8],allow:[3,5,7],alreadi:3,also:[4,6,7],alwai:[1,2,6,7,8],anaconda:6,ani:[0,1,2,4,7],annot:2,anoth:1,api:2,appear:[2,7],append_channel:2,appreci:6,appropri:[0,1],arbitrari:[0,3],arduou:2,argument:0,around:1,arrai:[0,1,2,3,7],arraylik:0,arraylike_read:[0,9],arraylikeread:1,artist:2,as_zx_bas:0,assign:[0,1],assum:[0,1,2,3],attach:2,attempt:[0,1,4],attribut:2,auto:5,avail:[0,1,4],b03:2,b6c1:2,back:[2,5],backend:1,badg:5,base:[0,1,2,3,4,5,7],becaus:1,befor:[0,1,5,7],begin:2,behavior:7,behind:4,being:[0,1,3],benchmark:5,benefit:4,best:4,better:4,between:3,bioformat:2,bit:[2,6],bitspersampl:2,black:5,bland:2,block:[1,5],blue:[2,4],bool:[0,1],bound:5,bowden:5,branch:[6,7],brown:5,bsd:7,bstczyx:0,buffer:[0,1],buffer_read:9,bufferediobas:[0,1],bufferread:0,bug:5,bugfix:[5,6],build:[5,6],bumpvers:5,bytearrai:1,cach:[4,7],call:[2,5,7],caller:2,can:[0,2,3,4,6,8],cannot:0,capabl:[0,1],care:4,cast:2,celllength:2,cellprofil:2,cellwidth:2,certain:0,cfg:5,chang:[2,6],channel:[0,1,2,3,5,7],channel_color:3,channel_count:2,channel_nam:3,channels_nam:[0,1],chart:4,chart_benchmark:4,cheapli:0,check:[0,5,6],checkout:6,child:0,choic:7,chunk:[1,7],chunk_by_dim:[0,1],classmethod:1,claus:7,clean:5,cleanup:5,clearli:4,client:[0,1,4],clone:[4,6,8],close:[0,1,3,5],cloud:4,cluster:[0,1,4,5],cluster_and_cli:[0,4],cmyk:2,code:[2,5,6,7],code_of_conduct:5,codecov:5,collect:[0,2],color:[2,3],column:2,columnnamingconvent:2,com:[4,6,8],command:[4,5,8],comment:5,commit:6,common:4,commun:0,compar:[2,4],compon:2,comput:[0,4,7],concurr:4,conda:4,configur:[0,4],conflict:0,conflictingargumentserror:0,connect:[0,1,4],consist:0,constant:[2,9],construct:[0,1,3,4],constructor:[0,2],contain:[0,1,2],content:9,context:[0,2,3,5],contribut:[5,7],convent:2,convert:3,cookiecutt:5,copi:[5,8],core:[0,1,4,6],could:1,count:2,cov:5,coverag:5,creat:[0,1,2,4,6],creation:3,credit:6,cron:5,curl:8,current:[0,1,2,4,7],cyx:[0,3],czi:[0,1,3,7],czi_read:[0,9],czifil:4,cziread:[0,1,3,5,7],czyx:[0,1,7],daca46f16346:2,dan:5,daniel:5,dask:[0,1,4,5,7],dask_cloudprovid:4,dask_data:[0,1,4,7],dask_jobqueu:4,dask_kwarg:0,dask_util:[4,9],data:[0,1,2,3,4,5,7],datatyp:2,date:2,datetim:2,deal:0,deepli:4,default_dim:0,default_read:[0,9],defaultord:0,defaultorderlist:0,defaultread:[0,1,7],delai:[0,4,5],delta:2,deltat:2,dep:5,depend:[1,7],deploy:4,depth:[0,1,2],deriv:2,derivedwrit:3,describ:7,descript:[1,2,5,6],desir:[0,1],detail:[0,1,6],determin:4,determine_read:0,dev:6,develop:[2,4,5,6],dict:[0,2],dictionari:[0,2],dictionary_index:2,differ:[1,4,7],dim:[0,1,3,5,7],dimens:[0,1,2,3,4,5,7],dimension:[0,2,7],dimension_ord:[3,5],dimensionord:2,dimi:5,dimitri:5,dir:4,direct:2,directli:0,disabl:[1,5],distribut:[0,1,4,5],do_:2,do_xytzc:2,do_xyzct:2,doc:[2,5],document:[2,4,5],doe:[3,5],doesn:[0,2,3,5],dom:2,don:[2,7,8],done:[3,6],doubl:5,down:[0,1],download:[4,6,8],download_test_resourc:[4,6],dtype:1,duck:1,ducktyp:2,durat:[0,2],dure:7,each:[0,1,2,3,4,7],easi:[2,4],edit:6,ef8af211:2,either:[0,2,4,8],element:[1,2],elementtre:[1,2],emp_bf:1,empti:0,enabl:[0,1,5],encod:2,encount:4,enough:[0,4],ensur:4,entir:[0,1,7],environ:[4,6],equival:7,error:[2,5],etc:[0,1,4],etre:[1,5],even:7,everi:[0,4,6],every_oth:[0,1],everyth:6,exactli:3,exampl:[0,1,2,3],except:9,exist:[2,3,5],expans:0,experi:[2,4],explor:2,exposur:2,exposuretim:2,extend:0,extens:0,extent:2,externaldescript:2,externalidentifi:2,extra:[0,1],f90f:2,fail:[1,5],fairli:4,fals:[0,1,3],fashion:4,fast:7,featur:[5,6],feedback:5,feel:4,field:0,fiff:5,fifth:0,file2:3,file3:3,file:[0,1,2,3,4,5,6,7],file_path:3,filelik:1,filenam:[0,2],filepath:0,fill:[0,1],fillord:2,filter:2,first:[0,1],first_and_last:[0,1],first_and_second:[0,1],first_thre:[0,1],firstc:2,firstt:2,firstz:2,fit:7,flag:3,flake8:5,fluo:1,fork:6,form:1,format:[1,2,3,5,7],found:[2,6,7],four:[2,4],fourth:[0,1],free:7,freebytecount:2,freeoffset:2,fresh:4,from:[0,1,2,4,5,7],full:[3,4],gain:4,gen:5,gener:[0,2,5],get:[0,1,2,4,5,7],get_acquisitiond:2,get_channel_count:2,get_channel_nam:[0,1,2,5,7],get_color:2,get_column:2,get_columnnamingconvent:2,get_deltat:2,get_descript:2,get_dimensionord:2,get_externaldescript:2,get_externalidentifi:2,get_firstc:2,get_firstt:2,get_firstz:2,get_float_attr:2,get_id:2,get_ifd:2,get_image_count:2,get_image_dask_data:[0,1,4,7],get_image_data:[0,1,4,7],get_image_descript:1,get_imageref:2,get_index:2,get_int_attr:2,get_n:2,get_nam:2,get_namespac:2,get_original_metadata_ref:2,get_original_metadata_valu:2,get_physical_pixel_s:[0,1],get_physicalsizei:2,get_physicalsizex:2,get_physicalsizez:2,get_pixel_typ:[1,2],get_pixeltyp:2,get_plane_count:2,get_planecount:2,get_planes_of_channel:2,get_positioni:2,get_positionx:2,get_positionz:2,get_row:2,get_rownamingconvent:2,get_sampl:2,get_samplesperpixel:2,get_sizec:2,get_sizei:2,get_sizet:2,get_sizex:2,get_sizez:2,get_statu:2,get_text:2,get_thec:2,get_thet:2,get_thez:2,get_timepoint:2,get_wel:2,get_well_nam:2,get_wellorigini:2,get_welloriginx:2,getter:5,gfp:1,git:[2,4,6,8],github:[4,5,6,7,8],given:[0,2,6],given_dim:0,grai:1,grayresponsecurv:2,grayresponseunit:2,greatli:6,green:[1,2],guess:0,guess_dim_ord:1,guid:8,handl:[0,1,2,6],has:[0,1,2,7],has_kei:2,has_original_metadata:2,hash:4,have:[0,1,2,4,5,7,8],help:6,here:6,higher:1,highli:7,hold:2,hook:2,hostcomput:2,how:[2,4,6],howev:1,hpc:4,html:[2,5],http:[2,4,8],huer:5,ibrahim:5,ids:2,ifd:2,imag:[0,1,2,3,4],image0:3,image2:3,image_count:2,image_nam:3,imagecodec:5,imageio:[1,4,7],imagelength:2,imagelik:[0,1],imageref:2,imagewidth:2,img1:0,img2:0,img40_1:2,img:[0,1,4,7],immedi:[1,4,7],implement:[4,7],imread:[0,4,5,7],imread_dask:[0,7],includ:[2,5,6],inconsistentpixeltyp:0,inconsistentshapeerror:0,increas:4,indent:2,index:[0,1,2,3,7],indic:2,indici:[0,1],infer:2,inform:7,initi:[0,2],inkset:2,input:0,insert:[2,3],insid:3,inspect:2,instal:[0,4,6],instanc:[0,2],instead:[1,7],intel_endian:0,intend:0,intent:0,interest:4,interfac:[0,1,2,3],interleav:2,intern:[4,7],interpret:2,invalid:[0,5],invaliddimensionorderingerror:0,invok:2,ioerror:3,is_om:1,is_this_typ:1,iso:2,item:2,iter:2,iter_original_metadata:2,iteritem:2,its:[2,5],itself:7,jackson:5,jacksonmaxfield:5,jami:5,just:7,kei:2,keyword:0,know:6,known:[0,5],known_dim:[0,5],kwarg:[0,1,3],label:5,larg:[1,4,7],last:0,latest:[5,7],lazi:0,lazy_data:7,lazy_s0t0:7,least:[2,4],left:0,leica:[1,5],less:2,let:[2,6],letter:[0,1,2],level:[1,2,5],librari:[1,2,4,7],lif:[1,5,7],lif_magic_byt:1,lif_memory_byt:1,lif_read:[0,9],lifread:1,like:2,link:5,lint:[5,6],linter:5,list:[0,1,2,7],littl:6,live:0,load:[0,1,3,4,7],load_slic:1,loadresult:0,local:[0,4,5,6],localclust:[0,1,4],localhost:0,locat:[2,3],log:5,look:2,lower:4,lxml:1,machin:0,made:2,madison:5,mai:7,main:2,maintain:6,major:4,make:[2,3,5,6,7],make_text_nod:2,makefil:5,manag:[0,3,4,5],mani:[2,4],map:1,master:[5,8],match:[0,2],matrix:1,matt:5,max:2,maxfield:5,maxsamplevalu:2,mean:[0,4],mechan:2,memori:[0,1,4,5,7],messag:0,meta:1,metadata:[0,1,2,3,4,5],method:[2,8],microscopi:[0,4,7],min:2,minor:4,minsamplevalu:2,minut:4,miss:0,mode:6,model:2,modif:2,modifi:2,modul:[5,7,9],more:[0,1,4],most:[2,8],motorola_endian:0,move:5,mpi_cmyk:2,multi:7,multipl:0,multiscal:5,must:0,mustafa:5,my_fil:[0,4,7],myimag:2,mymetadata:3,n_byte:0,name:[0,1,2,3,4,5,7],namespac:2,napari:[0,5],ndarrai:[0,1,3],necessari:3,need:[0,2,5,6],neglig:4,network:0,never:1,newlin:2,newplat:2,newsubfiletyp:2,node:2,non:[4,5],none:[0,1,2,3,5],normal:[4,7],note:[0,1,4],noth:3,now:6,npdtype:2,number:[0,2,4],numpi:[0,1,3,7],nworker:0,object:[0,2,3,4,7],off:[4,7],offset:5,often:2,old:5,om_:2,om_artist:2,om_bits_per_sampl:2,om_cell_length:2,om_cell_width:2,om_date_tim:2,om_document_nam:2,om_fill_ord:2,om_free_bytecount:2,om_free_offset:2,om_gray_response_curv:2,om_gray_response_unit:2,om_host_comput:2,om_image_length:2,om_image_width:2,om_ink_set:2,om_mak:2,om_max_sample_valu:2,om_min_sample_valu:2,om_model:2,om_new_subfile_typ:2,om_orient:2,om_page_numb:2,om_photometric_interpret:2,om_predictor:2,om_resolution_unit:2,om_samples_per_pixel:2,om_softwar:2,om_t4_opt:2,om_t6_opt:2,om_threshhold:2,om_tile_byte_count:2,om_tile_length:2,om_tile_offset:2,om_tile_width:2,om_transfer_funct:2,om_white_point:2,om_x_posit:2,om_x_resolut:2,om_y_posit:2,om_y_resolut:2,ome:[0,1,2,3,5],ome_metadata:3,ome_tiff_read:[0,9],ome_tiff_writ:[0,9],ome_xml:3,omemetadata:2,ometiffread:[0,1,7],ometiffwrit:3,omexml:[0,3,9],onc:[3,8],one:[0,1,2,4],onli:[0,2,4,7],open:[1,3],openmicroscopi:2,oper:[0,7],optim:[4,5],option:[0,1,2],orang:4,order:[0,1,2,3,5,7],org:2,organ:4,orient:2,origin:[2,6],originalmetadata:2,orphan:2,other:[0,2,3,4,6],our:[2,4,7],out:[0,1,2,3,4],out_orient:[0,1],output:[2,3],outsid:7,over:[2,3,5],overhead:4,overwrit:3,overwrite_fil:3,packag:[7,9],pad:0,page:[2,7],page_name_original_metadata:2,pagenumb:2,pair:2,parallel:[4,7],paramet:[0,1,3],parent:2,pars:[0,1,2,5],parser:7,particular:2,pass:[0,3,5,6,7],patch:5,path:[0,1,3],pathlib:[0,1,3],pathlik:3,pc_planar:2,per:2,perform:[3,4],person:4,photometr:2,physic:[0,1,2,3,5],physicalsizei:2,physicalsizex:2,physicalsizez:2,pi_cfa_arrai:2,pin:5,pip:[4,6,8],pixel:[0,1,2,3,5],pixels_physical_s:3,pixeltyp:2,planar:2,plane:[2,4],plane_count:2,planecount:2,plate:2,plate_id:2,platesducktyp:2,pleas:[4,6,7],plugin:7,png:[3,5],png_writer:[0,9],pngwriter:3,point:[3,5],populate_tiffdata:[2,5],portion:7,posit:2,positioni:2,positionx:2,positionz:2,possibl:7,pragmat:2,preconfigur:[0,1],predictor:2,prefer:[4,8],preload:[0,1],premad:3,prep:5,prepar:6,present:[0,1,2,4],pretti:[4,7],print:2,process:[0,1,4,8],programat:2,project:6,prone:2,proper:3,properli:[0,3,5],properti:[0,1,2,7],provid:[0,1,2,4],prs:5,prune:0,pt_:2,pt_uint8:2,publish:[4,5,6],pull:[5,6,7],pure:0,purpos:2,push:6,put:[0,3],py38:5,py39:5,pypi:5,pyramid:5,python:[0,1,2,4,5,6,7,8],qualifi:2,quickstart:5,quilt3distribut:5,quilt:4,quot:5,ran:4,rand:0,random:0,rang:[0,1],rapidli:2,raw:[0,4],read:[0,1,2,4,5,6],read_byt:0,read_uint16:0,read_uint32:0,read_uint64:0,reader:[0,3,4,5,7,9],readi:6,readlif:[1,5],readm:5,recent:8,recogn:0,recommend:[6,7],red:2,refactor:5,refer:[2,5],regardless:4,rel:2,relat:7,releas:[4,5,6,7],remind:6,remot:5,remov:[0,1,2,5],remove_channel:2,renam:5,replac:[2,5],replica:4,repo:[6,8],report:5,repositori:8,repres:[0,1,2],represent:2,request:[0,1,5,6,7],requir:5,reset:0,reshap:0,reshape_data:[0,1],resolutionunit:2,resolv:[5,6],resourc:[4,5,6],respect:5,result:7,retriev:[0,1,2,4],return_dim:0,revert:5,review:5,revis:5,rgb:[0,2,3,5],rgba:0,right:0,root:[1,2],root_nod:2,rootnod:2,rout:1,row:2,rownamingconvent:2,rst:5,run:[1,6,8],s0t0:7,s_1_t_1_c_10_z_20:[0,1],safe:4,sai:4,same:[0,1,3],sampl:2,samplesperpixel:2,save:3,save_slic:3,scene:[0,1,7],schema:2,script:[4,6],search:[2,7],second:[2,4],see:[0,1,2,4,7],seen:4,select:[0,1],self:[0,1],sequenc:5,seri:2,serializ:5,set:[1,2,5,6],set_acquisitiond:2,set_channel_count:2,set_color:2,set_column:2,set_columnnamingconvent:2,set_deltat:2,set_descript:2,set_dimensionord:2,set_externaldescript:2,set_externalidentifi:2,set_firstc:2,set_firstt:2,set_firstz:2,set_id:2,set_ifd:2,set_image_count:2,set_imageref:2,set_index:2,set_metadata:3,set_nam:2,set_physicalsizei:2,set_physicalsizex:2,set_physicalsizez:2,set_pixeltyp:2,set_plane_count:2,set_planecount:2,set_positioni:2,set_positionx:2,set_positionz:2,set_row:2,set_rownamingconvent:2,set_samplesperpixel:2,set_sizec:2,set_sizei:2,set_sizet:2,set_sizex:2,set_sizez:2,set_statu:2,set_text:2,set_thec:2,set_thet:2,set_thez:2,set_timepoint:2,set_wellorigini:2,set_welloriginx:2,setup:[4,5,8],shape:[0,1,7],sherman:5,should:[0,1,2,3,4,5],shuffl:0,shutdown:0,shutdown_cluster_and_cli:0,silent:3,similar:4,simpli:[0,1,4,7],simul:4,sinc:2,singl:4,site:2,situat:4,six:[0,1,7],size:[0,1,2,3,4,5,7],size_:[0,1],size_c:[0,1,3],size_i:[0,1,3],size_t:[0,1,3],size_x:[0,1,3],size_z:[0,1,3],sizec:2,sizei:2,sizet:2,sizex:2,sizez:2,slaton:5,slice:[0,1,3,7],slice_index:1,slurm:4,slurmclust:4,small:[4,7],softwar:[2,7],some:[2,7],sourc:[0,1,2,3],spatial:[0,1],spatiali:[0,1],spatialx:[0,1],spatialz:[0,1],spawn_cluster_and_cli:0,spec:5,specif:[0,1,2],specifi:[0,1,2,3,7],sphinx:5,split:2,split_qn:2,spw:2,src:2,stack_count:2,stage:2,standard:[0,1,5],statu:2,stc:7,stczyx:[0,1,3,7],storag:1,store:[1,2,7],str:[0,1,3],strategi:4,strictli:1,string:[0,1,2,5,7],structur:2,structured_annot:2,structuredannot:2,stzcyx:3,subclass:0,submit:6,submodul:9,subpackag:9,subsequ:0,suppli:2,support:[0,2,4,5,7],supported_read:0,sure:6,swap:4,sylvain:5,t4option:2,t6option:2,tag:[2,6],tag_nam:2,take:[0,3],tarbal:8,task:5,tcp:0,tczyx:0,team:[4,5],templat:5,ten:0,tend:7,termin:8,test:[5,6],test_ome_tiff_writ:5,text:2,than:[2,4],thec:2,thei:6,them:[0,2,3],thet:2,thez:2,thi:[0,1,2,3,4,8],thing:2,those:7,thread:4,three:2,threshhold:2,through:[6,8],thrown:0,tie:2,tif:[2,3],tiff:[0,1,2,3,4,5,6,7],tiff_read:[0,9],tiffdata:2,tifffil:[1,4],tiffread:[0,1,7],tilebytecount:2,tilelength:2,tileoffset:2,tilewidth:2,time:[0,2,4,7],timepoint:[0,2],timepoint_count:2,to_xml:2,todo:3,toloudi:5,too:7,tool:5,top:[2,4,5],transferfunct:2,transform:[1,9],transpose_to_dim:0,tree:1,trigger:5,tupl:[0,1,7],two:[2,3,4],type:[1,2,3,7,9],typic:[2,4],tzyx:0,uint16:1,uint32:1,uint8:1,unabl:2,underli:[1,4],understand:4,unfortun:4,unicod:2,unifi:[0,1],union:[0,1,3],unit:[2,5],unsupportedfileformaterror:0,until:7,updat:[5,7],upload:6,upper:5,urn:2,use:[0,1,2,3,5,7],use_dask:1,used:[0,1,2],useful:7,user:[0,7],uses:1,using:[0,1,2,4],utf:2,util:[4,7],uuid:2,valid:0,valu:[1,2,3],vari:[0,2],variabl:0,variou:4,vbtcxzy:0,vendor:[0,9],version:[5,6],view:[2,4],view_imag:7,view_napari:[0,7],viewer:0,virtualenv:6,visit:7,wai:[2,4],wait:4,want:7,warn:[0,5],websit:6,welcom:6,well:[0,1,2,4],well_id:2,well_nod:2,wellorigini:2,welloriginx:2,wellsampl:2,wellsample_id:2,wellsampleducktyp:2,wellsducktyp:2,were:5,what:4,when:[0,1,4,6,7],where:2,which:[0,1,2,4],whichev:4,whitepoint:2,wish:4,with_tim:0,within:[2,3],without:0,word:7,work:[4,6],worker:4,workflow:5,workload:4,workstat:4,would:0,wrap:[1,2,5],wrapper:1,write:[2,3,7],writer2:3,writer3:3,writer:[0,5,9],written:3,www:2,xml:[1,2,3,5],xposit:2,xresolut:2,xsd:2,xsd_now:2,xytzc:2,you:[2,4,6,7,8],your:[4,6,7,8],your_development_typ:6,your_name_her:6,yourself:[2,4],yposit:2,yresolut:2,yzx:0,zeiss_10byt:1,zeiss_2byt:1,zero:2,zisrawfil:1,zstack_t8:0,zstack_t8_data:0,zyx:[0,1]},titles:["aicsimageio package","aicsimageio.readers package","aicsimageio.vendor package","aicsimageio.writers package","Benchmarks","Changelog","Contributing","Welcome to aicsimageio\u2019s documentation!","Installation","aicsimageio"],titleterms:{"new":5,aics_imag:0,aicsimageio:[0,1,2,3,7,9],arraylike_read:1,benchmark:4,buffer_read:0,changelog:5,config:4,consider:7,constant:0,content:[0,1,2,3],contribut:6,czi_read:1,dask_util:0,default_read:1,delai:7,deploi:6,develop:7,discuss:4,document:7,except:0,featur:7,fix:5,from:8,full:7,get:6,histor:4,imag:7,indic:7,instal:8,interact:7,lif_read:1,metadata:7,modul:[0,1,2,3],napari:7,note:7,ome_tiff_read:1,ome_tiff_writ:3,omexml:2,other:5,packag:[0,1,2,3],perform:7,png_writer:3,quick:7,read:7,reader:1,releas:8,result:4,run:4,sourc:8,stabl:8,start:[6,7],submodul:[0,1,2,3],subpackag:0,tabl:7,test:4,thi:7,tiff_read:1,transform:0,type:0,vendor:2,version:7,viewer:7,welcom:7,writer:3}})