import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine
from rasterio.plot import show
from rasterio.mask import mask
from shapely import geometry
from shapely.geometry import box, MultiPolygon
import numpy
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
dmp_nodata_val = dmp.nodata
dmt_nodata_val = dmt.nodata


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

def create_masking_matrix(dmp_matrix, dmt_matrix, dmp_nodata_val, dmt_nodata_val):
    masking_matrix = numpy.where((abs(dmp_matrix[0]-dmt_matrix[0])<THRESHOLD) & (dmt_matrix[0] != dmt_nodata_val) & (dmp_matrix[0] != dmp_nodata_val), 1, 0)
    return masking_matrix


print(create_masking_matrix(dmp_intersect_matrix, dmt_intersect_matrix, dmp_nodata_val, dmt_nodata_val))


#print(bb_dmp)
print(type(dmp_nodata_val))
#print(type(dmt_intersect_matrix[0]))
#print(dmp_intersect_matrix)
#print(dmt_intersect_matrix)
print(intersection)
print(type(dmp))
print(type(bb_dmp))
print(type(intersection))
#show(bb_dmp)
#show(dmt)