import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine
from rasterio.plot import show
from rasterio.mask import mask
from shapely import geometry
from shapely.geometry import box, MultiPolygon
'''
with rasterio.open("DSM_1M_Clip.tif") as dsm:
    kwargs = dsm.meta
    kwargs.update(driver="GTiff", dtype=rasterio.float32,count=1,compress="lzw")
    #print(kwargs)

    for ji, window in dsm.block_windows(1):
        x = dsm.read(1, window=window)
        print (x.shape)
'''

dmp = rasterio.open("DSM_1M_Clip.tif")
dmt = rasterio.open("DTM_Clip_3.tif")

bb_dmp = box(dmp.bounds[0], dmp.bounds[1], dmp.bounds[2], dmp.bounds[3])
bb_dmt = box(dmt.bounds[0], dmt.bounds[1], dmt.bounds[2], dmt.bounds[3])

xminDMP, yminDMP, xmaxDMP, ymaxDMP = dmp.bounds
xminDMT, yminDMT, xmaxDMT, ymaxDMT = dmt.bounds

intersection = bb_dmp.intersection(bb_dmt)

intersection_to_multipolyg = [intersection]
poly = geometry.MultiPolygon(intersection_to_multipolyg)

dmp_intersect_matrix = rasterio.mask.mask(dmp, poly, crop=True)
dmt_intersect_matrix = rasterio.mask.mask(dmt, poly, crop=True)

THRESHOLD = 1 # meters

#print(bb_dmp)
#print(type(dmp_intersect_matrix[0]))
#print(type(dmt_intersect_matrix[0]))
print(dmp_intersect_matrix)
print(dmt_intersect_matrix)
print(intersection)
print(type(dmp))
print(type(bb_dmp))
print(type(intersection))
#show(bb_dmp)
#show(dmt)