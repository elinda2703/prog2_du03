import rasterio
import rasterio.crs
import rasterio.transform
import rasterio.errors
from rasterio.windows import Window
from rasterio.transform import Affine
from rasterio.mask import mask
from rasterio.crs import CRS
from shapely import geometry
from shapely.geometry import box, MultiPolygon
import numpy as np
from matplotlib import pyplot as plt
import argparse, sys


def get_raster_intersections(raster1, raster2):
    bb_raster1 = box(raster1.bounds[0], raster1.bounds[1], raster1.bounds[2], raster1.bounds[3])
    bb_raster2 = box(raster2.bounds[0], raster2.bounds[1], raster2.bounds[2], raster2.bounds[3])
    
    xminR1, yminR1, xmaxR1, ymaxR1 = raster1.bounds
    xminR2, yminR2, xmaxR2, ymaxR2 = raster2.bounds
    
    intersection = bb_raster1.intersection(bb_raster2)
    
    transform = Affine(raster1.res[0], 0.0, intersection.bounds[0], 0.0, -raster1.res[1], intersection.bounds[3])
    
    p1Y = intersection.bounds[3] - raster1.res[1]/2
    p1X = intersection.bounds[0] + raster1.res[0]/2
    p2Y = intersection.bounds[1] + raster1.res[1]/2
    p2X = intersection.bounds[2] - raster1.res[0]/2
    
    row1R1 = int((ymaxR1 - p1Y)/raster1.res[1])
    row1R2 = int((ymaxR2 - p1Y)/raster2.res[1])
    col1R1 = int((p1X - xminR1)/raster1.res[0])
    col1R2 = int((p1X - xminR2)/raster1.res[0])

    row2R1 = int((ymaxR1 - p2Y)/raster1.res[1])
    row2R2 = int((ymaxR2 - p2Y)/raster2.res[1])
    col2R1 = int((p2X - xminR1)/raster1.res[0])
    col2R2 = int((p2X - xminR2)/raster1.res[0])

    
    width1 = col2R1 - col1R1 + 1
    width2 = col2R2 - col1R2 + 1
    height1 = row2R1 - row1R1 + 1
    height2 = row2R2 - row1R2 + 1
    
    raster1_intersection = raster1.read(1, window=Window(col1R1, row1R1, width1, height1))
    raster2_intersection = raster2.read(1, window=Window(col1R2, row1R2, width2, height2))
    #intersection_polygon = geometry.MultiPolygon([intersection])
    #raster1_intersection = rasterio.mask.mask(raster1, intersection_polygon, crop=True)
    #raster2_intersection = rasterio.mask.mask(raster2, intersection_polygon, crop=True)
    #print(intersection_polygon)
    return raster1_intersection, raster2_intersection, transform

def matrix_size(matrix):
    height = matrix.shape[0]
    width = matrix.shape[1]
    return height, width

def create_masking_matrix(dmp_matrix, dmt_matrix, dmp_nodata_val, dmt_nodata_val):
    THRESHOLD = 1 # meters
    masking_matrix = np.where((abs(dmp_matrix-dmt_matrix) < THRESHOLD) & (dmt_matrix != dmt_nodata_val) & (dmp_matrix != dmp_nodata_val), 1, np.nan)
    return masking_matrix#, dmp_matrix[1]

def extract_terrain(dmp_matrix, masking_matrix):
    extraction = np.where(masking_matrix == 1, dmp_matrix, np.nan)
    return extraction#, masking_matrix[1]

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
 
            try:

                if dmp.crs == dmt.crs:
                    pass
                
                else:
                    sys.exit("Zvolene rastery nemaji zhodny souradnicovy system, nebo je v nich ruzne zadefinovany. Nelze provest dalsi vypocty.")
            
            except rasterio.errors.CRSError(dmp):
                sys.exit("Ve zvolenem surface rasteru neni definovany validny souradnicovy system.")
                
            except rasterio.errors.CRSError(dmt):
                sys.exit("Ve zvolenem terrain rasteru neni definovany validny souradnicovy system.")  
            
            except rasterio.errors.RasterioError():
                sys.exit("Neco se nepovedlo, program se ted ukonci.")
            
            
            
                
            dmp_intersection, dmt_intersection, intersection_transform = get_raster_intersections(dmp, dmt)
            #intersection1, intersection2 = get_raster_intersections(dmp, dmt)

            intersection_height, intersection_width = matrix_size(dmp_intersection)
            #for ji, window in dmp.block_windows(1):
                #r = dmp.read(1, window=window)
                
            #for ji, window in dmt.block_windows(1):
                #f = dmt.read(1, window=window)
            
            #print(r)
            #print(f)
            mask_dataset = create_masking_matrix(dmp_intersection, dmt_intersection, dmp.nodata, dmt.nodata)
            
            terrain_dataset = extract_terrain(dmp_intersection, mask_dataset)

            slope_deg = create_slope_matrix(terrain_dataset)

            #slope_height, slope_width = matrix_size(slope_deg)

            
            with rasterio.open(
                "mask.tiff",
                "w",
                driver = "GTiff",
                height = intersection_height, 
                width = intersection_width, 
                count = 1, 
                nodata = np.nan, 
                dtype = dmp.meta["dtype"], 
                crs = dmp.crs, 
                transform = intersection_transform) as mask_export:
                    mask_export.write(mask_dataset, 1)
            
                        #dmpwin = dmp.read(1, window=window).astype(float)
                        #dmtwin = dmt.read(1, window=window).astype(float)
                        #mask = create_masking_matrix(dmp_intersection, dmt_intersection, dmp.nodata, dmt.nodata)

                        #mask_export.write(mask[0])
                        #mask_export.write_band(1, mask[0][0].astype(rasterio.float32), window=)
                # CRS SETTING OPRAVIT

            with rasterio.open(
                "slopes.tiff", 
                "w",
                driver = "GTiff",
                height = intersection_height,
                width = intersection_width,
                count = 1,
                nodata = np.nan,
                dtype = dmp.meta["dtype"],
                crs = dmp.crs,
                transform = intersection_transform) as slope_export:
                    slope_export.write(slope_deg, 1)

                    
                        #dmpwin = dmp.read(1, window=window).astype(float)
                        #dmtwin = dmt.read(1, window=window).astype(float)
                        #mask = create_masking_matrix(dmp_intersection, dmt_intersection, dmp.nodata, dmt.nodata)
                        #terrain = extract_terrain(dmpwin, mask)
                        #slopes = create_slope_matrix(terrain[0][0])
                        #slope_export.write_band(1, slopes.astype(rasterio.float32), window=Window(inter_coords[0][0],inter_coords[0][1],inter_coords[0][2],inter_coords[0][3]))

            

            visualize_rasters(mask_dataset,terrain_dataset,slope_deg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False, description = 'Skript, ktery ze vstupniho DMT a DMP\
        vytvori rastery sklonu nezastavenych ploch a nalezeny nezastavene plochy.')
    parser.add_argument('--terrain', action = "store", dest = "terraininput", required = True, help = 'Cesta k DMT ve formatu .tif.')
    parser.add_argument('--surface', action = "store", dest = "surfaceinput", required = True, help = 'Cesta k DMP ve formatu .tif.')
    args = parser.parse_args()
    proceed(args.surfaceinput, args.terraininput)