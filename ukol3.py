import rasterio
import rasterio.crs
import rasterio.transform
from rasterio.windows import Window
from rasterio.transform import Affine
from rasterio.mask import mask
from rasterio.crs import CRS
from shapely import geometry
from shapely.geometry import box, MultiPolygon
import numpy as np
from matplotlib import pyplot as plt
import argparse


def get_raster_intersections(raster1, raster2):
    bb_raster1 = box(raster1.bounds[0], raster1.bounds[1], raster1.bounds[2], raster1.bounds[3])
    bb_raster2 = box(raster2.bounds[0], raster2.bounds[1], raster2.bounds[2], raster2.bounds[3])
    intersection = bb_raster1.intersection(bb_raster2)
    intersection_polygon = geometry.MultiPolygon([intersection])
    raster1_intersection = rasterio.mask.mask(raster1, intersection_polygon, crop=True)
    raster2_intersection = rasterio.mask.mask(raster2, intersection_polygon, crop=True)
    return raster1_intersection, raster2_intersection

def matrix_size(matrix):
    height = matrix.shape[0]
    width = matrix.shape[1]
    return height, width

def create_masking_matrix(dmp_matrix, dmt_matrix, dmp_nodata_val, dmt_nodata_val):
    THRESHOLD = 1 # meters
    masking_matrix = np.where((abs(dmp_matrix[0]-dmt_matrix[0]) < THRESHOLD) & (dmt_matrix[0] != dmt_nodata_val) & (dmp_matrix[0] != dmp_nodata_val), 1, np.nan)
    return masking_matrix, dmp_matrix[1]

def extract_terrain(dmp_matrix, masking_matrix):
    extraction = np.where(masking_matrix[0] == 1, dmp_matrix[0], np.nan)
    return extraction, masking_matrix[1]

def create_slope_matrix (terrain_matrix):
    px, py = np.gradient(terrain_matrix, 1)
    slope = np.sqrt(px ** 2 + py ** 2)
    slope_deg = np.degrees(np.arctan(slope))
    return slope_deg

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

def proceed(surfaceinput, terraininput):
    with rasterio.open(surfaceinput) as dmp:

        with rasterio.open(terraininput) as dmt:
 
            dmp_meta = dmp.meta
            dmt_meta = dmt.meta

            if dmp.crs == dmt.crs:
                print('OK')
                
            else:
                print("nn")

            
            dmp_intersection, dmt_intersection = get_raster_intersections(dmp, dmt)
            #intersection1, intersection2 = get_raster_intersections(dmp, dmt)

            mask_height, mask_width = matrix_size(dmp_intersection[0][0])
            
            #mask_dataset = create_masking_matrix(dmp_intersection, dmt_intersection, dmp.nodata, dmt.nodata)
            
            #terrain_dataset = extract_terrain(dmp_intersection, mask_dataset)

            #slope_deg = create_slope_matrix(terrain_dataset[0][0])

            #slope_height, slope_width = matrix_size(slope_deg)

            with rasterio.open(
                "mask.tiff",
                "w",
                driver = "GTiff",
                height = mask_height, 
                width = mask_width, 
                count = 1, 
                nodata = np.nan, 
                dtype = dmp.meta["dtype"], 
                crs = dmp.crs, 
                transform = dmp_intersection[1]) as mask_export:

                    for ji, window in dmp.block_windows(1):
                        print(ji, window)
                        #dmpwin = dmp.read(1, window=window).astype(float)
                        #dmtwin = dmt.read(1, window=window).astype(float)
                        mask = create_masking_matrix(dmp_intersection, dmt_intersection, dmp.nodata, dmt.nodata)

                        #mask_export.write(mask[0])
                        mask_export.write_band(1, mask[0][0].astype(rasterio.float32), window=window)
                # CRS SETTING OPRAVIT

            with rasterio.open(
                "slopes.tiff", 
                "w",
                driver = "GTiff",
                height = mask_height,
                width = mask_width,
                count = 1,
                nodata = np.nan,
                dtype = dmp.meta["dtype"],
                crs = dmp.crs,
                transform = dmp_intersection[1]) as slope_export:

                    for ji, window in dmp.block_windows(1):
                        dmpwin = dmp.read(1, window=window).astype(float)
                        #dmtwin = dmt.read(1, window=window).astype(float)
                        mask = create_masking_matrix(dmp_intersection, dmt_intersection, dmp.nodata, dmt.nodata)
                        terrain = extract_terrain(dmpwin, mask)
                        slopes = create_slope_matrix(terrain[0][0])
                        slope_export.write_band(1, slopes.astype(rasterio.float32), window=window)

            

            visualize_rasters(mask[0][0,:,:],terrain[0][0],slopes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False, description = 'Skript, ktery ze vstupniho DMT a DMP\
        vytvori rastery sklonu nezastavenych ploch a nalezeny nezastavene plochy.')
    parser.add_argument('--terrain', action = "store", dest = "terraininput", required = True, help = 'Cesta k DMT ve formatu .tif.')
    parser.add_argument('--surface', action = "store", dest = "surfaceinput", required = True, help = 'Cesta k DMP ve formatu .tif.')
    args = parser.parse_args()
    proceed(args.surfaceinput, args.terraininput)