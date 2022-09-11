import rasterio
import rasterio.crs
from rasterio.windows import Window
from rasterio.transform import Affine
from rasterio.plot import show
from rasterio.mask import mask
from rasterio.crs import CRS
from shapely import geometry
from shapely.geometry import box, MultiPolygon
import numpy
from copy import deepcopy

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
dmp_meta = dmp.meta
dmt_meta = dmt.meta


if dmp_meta['crs'] == dmt_meta['crs']:
    print('OK')

else:
    print("nn")

bb_dmp = box(dmp.bounds[0], dmp.bounds[1], dmp.bounds[2], dmp.bounds[3])
bb_dmt = box(dmt.bounds[0], dmt.bounds[1], dmt.bounds[2], dmt.bounds[3])

intersection = bb_dmp.intersection(bb_dmt)

intersection_to_multipolyg = [intersection]
poly = geometry.MultiPolygon(intersection_to_multipolyg)

dmp_intersect_matrix = rasterio.mask.mask(dmp, poly, crop=True)
dmt_intersect_matrix = rasterio.mask.mask(dmt, poly, crop=True)

mask_height = dmp_intersect_matrix[0][0].shape[0]
mask_width = dmp_intersect_matrix[0][0].shape[1]

THRESHOLD = 1 # meters


def create_masking_matrix(dmp_matrix, dmt_matrix, dmp_nodata_val, dmt_nodata_val):
    disgusting_matrix = deepcopy(dmp_matrix)
    masking_matrix = numpy.where((abs(dmp_matrix[0]-dmt_matrix[0]) < THRESHOLD) & (dmt_matrix[0] != dmt_nodata_val) & (dmp_matrix[0] != dmp_nodata_val), 1, 0)
    disgusting_matrix[0][0] = masking_matrix
    return disgusting_matrix

pain=create_masking_matrix(dmp_intersect_matrix, dmt_intersect_matrix, dmp_nodata_val, dmt_nodata_val)

with rasterio.open("finished_mask.tif","w",driver = "GTiff",height=mask_height, width=mask_width, count=1, nodata=0, dtype=dmp.meta["dtype"], crs=CRS.from_wkt('LOCAL_CS["S-JTSK_Krovak_East_North",UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","5514"]]'), transform=pain[1]) as peepeepoopoo:
    peepeepoopoo.write(pain[0])



#print(create_masking_matrix(dmp_intersect_matrix, dmt_intersect_matrix, dmp_nodata_val, dmt_nodata_val))
print(CRS.from_wkt('LOCAL_CS["S-JTSK_Krovak_East_North",UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","5514"]]'))
#print((dmp_meta["crs"]))
#print(type(peepeepoopoo))
#print(pain)
#print(create_masking_matrix(dmp_intersect_matrix, dmt_intersect_matrix, dmp_nodata_val, dmt_nodata_val))
#print(dmp_meta.GetAttrValue('AUTHORITY',1))
#print(bb_dmp)
#print(type(dmp_nodata_val))
#print(type(dmt_intersect_matrix[0]))
#print(dmp_intersect_matrix[0])
#print(dmt_intersect_matrix)
#print(intersection)
#print(type(dmp))
#print(type(bb_dmp))
#print(type(intersection))
#show(bb_dmp)
#print(dmp.meta["dtype"])
#print(dmp_intersect_matrix[0][0].shape[0])
#print(dmp_intersect_matrix[0][0].shape[1])
#print(dmt_intersect_matrix[0][0].shape[0])
#print(dmt_intersect_matrix[0][0].shape[1])
#print(dmp_intersect_matrix[0][0])
#print(dmt_meta)

"""
. maska +- done
. pridat prostorove informace (nebo si precist dokumentaci for once a neztratit prostorova data), exportovat raster
. aplikovat na puvodni rastr, odstranit nepotrebne pixely (numpy.where?)
. pomoci numpy udelat slope, pripadne pridat prostorova data, exporotvat raster
. pohrat si s blokama
. zadani parametrem
"""