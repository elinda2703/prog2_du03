import rasterio
import rasterio.crs
from rasterio.windows import Window
from rasterio.transform import Affine
from rasterio.plot import show
from rasterio.mask import mask
from rasterio.crs import CRS
from shapely import geometry
from shapely.geometry import box, MultiPolygon
import numpy as np
from matplotlib import pyplot as plt

'''
with rasterio.open("DSM_1M_Clip.tif") as dsm:
    kwargs = dsm.meta
    kwargs.update(driver="GTiff", dtype=rasterio.float32,count=1,compress="lzw")
    #print(kwargs)

    for ji, window in dsm.block_windows(1):
        x = dsm.read(1, window=window)
        print (x.shape)
'''

with rasterio.open("DSM_1M_Clip.tif") as dmp:
    
    with rasterio.open("DTM_Clip_3.tif") as dmt:
        
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
            masking_matrix = np.where((abs(dmp_matrix[0]-dmt_matrix[0]) < THRESHOLD) & (dmt_matrix[0] != dmt_nodata_val) & (dmp_matrix[0] != dmp_nodata_val), 1, np.nan)
            return masking_matrix, dmp_matrix[1]

        pain=create_masking_matrix(dmp_intersect_matrix, dmt_intersect_matrix, dmp_nodata_val, dmt_nodata_val)

        with rasterio.open(
            "finished_mask.tif",
            "w",
            driver = "GTiff",
            height = mask_height, 
            width = mask_width, 
            count = 1, 
            nodata = np.nan, 
            dtype = dmp.meta["dtype"], 
            crs = dmp_meta['crs'], 
            transform = pain[1]) as peepeepoopoo:
                peepeepoopoo.write(pain[0])
            # CRS SETTING OPRAVIT

        def extract_terrain(dmp_matrix, masking_matrix):
            extraction = np.where(masking_matrix[0] == 1, dmp_matrix[0], np.nan)
            return extraction, masking_matrix[1]

        agony = extract_terrain(dmp_intersect_matrix, pain)

        with rasterio.open(
            "surface.tif", 
            "w",
            driver = "GTiff",
            height = mask_height,
            width = mask_width,
            count = 1,
            nodata = np.nan,
            dtype = dmp.meta["dtype"],
            crs = dmp_meta['crs'],
            transform = agony[1]) as society:
                society.write(agony[0])

        px, py = np.gradient(agony[0][0], 1)
        slope = np.sqrt(px ** 2 + py ** 2)
        slope_deg = np.degrees(np.arctan(slope))
        slope_height=slope_deg.shape[0]
        slope_width=slope_deg.shape[1]
        #print(agony[0][0])

        print(slope_height)
        print(slope_width)

        with rasterio.open(
            "slopes.tif", 
            "w",
            driver = "GTiff",
            height = slope_height,
            width = slope_width,
            count = 1,
            nodata = np.nan,
            dtype = dmp.meta["dtype"],
            crs = dmp_meta['crs'],
            transform = agony[1]) as garbage:
                garbage.write(slope_deg, 1)

        #print(create_masking_matrix(dmp_intersect_matrix, dmt_intersect_matrix, dmp_nodata_val, dmt_nodata_val))
        #print(CRS.from_wkt('LOCAL_CS["S-JTSK_Krovak_East_North",UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","5514"]]'))
        #print(pain[1])
        #print(type(peepeepoopoo))
        #print(intersection)
        #print(poly)
        #print(mask_height)
        #print(mask_width)
        #print(create_masking_matrix(dmp_intersect_matrix, dmt_intersect_matrix, dmp_nodata_val, dmt_nodata_val))
        #print(dmp_meta.GetAttrValue('AUTHORITY',1))
        #print(bb_dmp)
        #print(type(dmp_nodata_val))
        #print(type(dmt_intersect_matrix[0]))
        #print(dmp_intersect_matrix[0])
        #plt.imshow(full_dmp[0,:,:])
        """
        figure, axis = plt.subplots(2, 2)

        #plt.figure(1)
        plt.imshow(pain[0][0,:,:])
        plt.colorbar()
        #plt.figure(2)
        plt.imshow(agony[0][0])
        plt.colorbar()
        #plt.figure(3)
        plt.imshow(slope_deg)
        plt.colorbar()   
        plt.show()
        """
        fig, axis = plt.subplots(1, 3)
        fig.suptitle("Rasters")
        axis[0].imshow(pain[0][0,:,:])
        axis[1].imshow(agony[0][0])
        axis[2].imshow(slope_deg)
        axis[0].set_title("Elevation")
        axis[1].set_title("Mask")
        axis[2].set_title("Slope")
        plt.show()
        #plt.imshow(dmp_intersect_matrix[0])
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
        vycistit kod
        nastavit nodata value
        """

