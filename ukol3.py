import rasterio
import rasterio.crs
import rasterio.transform
from rasterio.windows import Window
from rasterio.transform import Affine
from rasterio.plot import show
from rasterio.mask import mask
from rasterio.crs import CRS
from shapely import geometry
from shapely.geometry import box, MultiPolygon
import numpy as np
from matplotlib import pyplot as plt
import sys, getopt

'''
with rasterio.open("DSM_1M_Clip.tif") as dsm:
    kwargs = dsm.meta
    kwargs.update(driver="GTiff", dtype=rasterio.float32,count=1,compress="lzw")
    #print(kwargs)

    for ji, window in dsm.block_windows(1):
        x = dsm.read(1, window=window)
        print (x.shape)
'''
def get_raster_intersections (dmp, dmt):
    bb_dmp = box(dmp.bounds[0], dmp.bounds[1], dmp.bounds[2], dmp.bounds[3])
    bb_dmt = box(dmt.bounds[0], dmt.bounds[1], dmt.bounds[2], dmt.bounds[3])
    intersection = bb_dmp.intersection(bb_dmt)
    poly = geometry.MultiPolygon([intersection])
    dmp_intersect_matrix = rasterio.mask.mask(dmp, poly, crop=True)
    dmt_intersect_matrix = rasterio.mask.mask(dmt, poly, crop=True)
    return dmp_intersect_matrix, dmt_intersect_matrix

def matrix_size (matrix):
    height = matrix.shape[0]
    width = matrix.shape[1]
    return height, width

def proceed(surfaceinput, terraininput):
    with rasterio.open(surfaceinput) as dmp:

        with rasterio.open(terraininput) as dmt:
 
            dmp_nodata_val = dmp.nodata
            dmt_nodata_val = dmt.nodata
            dmp_meta = dmp.meta
            dmt_meta = dmt.meta

            if dmp_meta['crs'] == dmt_meta['crs']:
                print('OK')
                
            else:
                print("nn")

            


            dmp_intersect_matrix, dmt_intersect_matrix = get_raster_intersections(dmp,dmt)



            mask_height, mask_width = matrix_size(dmp_intersect_matrix[0][0])
            THRESHOLD = 1 # meters


            def create_masking_matrix(dmp_matrix, dmt_matrix, dmp_nodata_val, dmt_nodata_val):
                masking_matrix = np.where((abs(dmp_matrix[0]-dmt_matrix[0]) < THRESHOLD) & (dmt_matrix[0] != dmt_nodata_val) & (dmp_matrix[0] != dmp_nodata_val), 1, np.nan)
                return masking_matrix, dmp_matrix[1]

            mask_dataset=create_masking_matrix(dmp_intersect_matrix, dmt_intersect_matrix, dmp_nodata_val, dmt_nodata_val)

            with rasterio.open(
                "mask.tif",
                "w",
                driver = "GTiff",
                height = mask_height, 
                width = mask_width, 
                count = 1, 
                nodata = np.nan, 
                dtype = dmp.meta["dtype"], 
                crs = dmp_meta['crs'], 
                transform = mask_dataset[1]) as mask_export:
                    mask_export.write(mask_dataset[0])
                # CRS SETTING OPRAVIT

            def extract_terrain(dmp_matrix, masking_matrix):
                extraction = np.where(masking_matrix[0] == 1, dmp_matrix[0], np.nan)
                return extraction, masking_matrix[1]

            terrain_dataset = extract_terrain(dmp_intersect_matrix, mask_dataset)

            '''
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
                transform = terrain_dataset[1]) as terrain_export:
                    terrain_export.write(terrain_dataset[0])
            '''
            def create_slope_matrix (terrain_matrix):
                px, py = np.gradient(terrain_matrix, 1)
                slope = np.sqrt(px ** 2 + py ** 2)
                slope_deg = np.degrees(np.arctan(slope))
                return slope_deg

            slope_deg=create_slope_matrix(terrain_dataset[0][0])

            slope_height=slope_deg.shape[0]
            slope_width=slope_deg.shape[1]


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
                transform = terrain_dataset[1]) as slope_export:
                    slope_export.write(slope_deg, 1)

            def visualize_rasters (mask_raster, terrain_raster, slope_raster):
                fig, axis = plt.subplots(1, 3)
                fig.suptitle("Rasters")
                axis[0].imshow(mask_raster)
                axis[1].imshow(terrain_raster)
                axis[2].imshow(slope_raster)
                axis[0].set_title("Mask")
                axis[1].set_title("Extracted altitudes")
                axis[2].set_title("Slope")
                plt.show()

            visualize_rasters(mask_dataset[0][0,:,:],terrain_dataset[0][0],slope_deg)
            """
            . pohrat si s blokama
            . zadani parametrem
            vycistit kod
            """

def main(argv):

    terraininput = ""
    surfaceinput = ""

    try:
        opts, args = getopt.getopt(argv, "t:s:", ["terrain=", "surface="])
    
    except getopt.GetoptError:
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("--terrain"):
            terraininput = arg
        elif opt in ("--surface"):
            surfaceinput = arg
    
    print(f"Input terrain: {terraininput}")
    print(f"Input surface: {surfaceinput}")
    proceed(surfaceinput, terraininput)

if __name__ == "__main__":
    main(sys.argv[1:])