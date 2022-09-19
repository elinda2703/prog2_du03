try:
    import rasterio
    import rasterio.crs
    import rasterio.transform
    import rasterio.errors
    from rasterio.windows import Window
    from rasterio.transform import Affine
    from shapely.geometry import box
    import numpy as np
    import argparse, sys

except ModuleNotFoundError:
    sys.exit("Jedna z pozadovanych knihoven neni nainstalovana.")
    
except ImportError:
    sys.exit("Jedna z pozadovanych knihoven neni nainstalovana.")

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
    
    r1win = raster1.read(1, window=Window(col1R1, row1R1, width1, height1))
    r2win = raster2.read(1, window=Window(col1R2, row1R2, width2, height2))
    
    return r1win, r2win, transform
    
        #raster_intersection_list = [r1win, r2win]
    #intersection_polygon = geometry.MultiPolygon([intersection])
    #raster1_intersection = rasterio.mask.mask(raster1, intersection_polygon, crop=True)
    #raster2_intersection = rasterio.mask.mask(raster2, intersection_polygon, crop=True)
    #print(intersection_polygon)
    #return raster_intersection_list, transform

def process_by_blocks(raster1, raster2, mask_export, slope_export, 
                      raster_width, raster_height, raster1_nodata, raster2_nodata):
    block_size = 128
    slices = [(col_start, row_start, block_size, block_size) \
                for col_start in list(range(0, raster_width, 128)) \
                for row_start in list(range(0, raster_height, 128))
    ]
    for slc in slices:

        current_block_r1 = raster1[(slc[1]):(slc[1] + block_size), slc[0]:(slc[0] + block_size)]
        current_block_r2 = raster2[(slc[1]):(slc[1] + block_size), slc[0]:(slc[0] + block_size)]
        
        mask_dataset = create_masking_matrix(current_block_r1, current_block_r2, raster1_nodata, raster2_nodata)
        terrain_dataset = extract_terrain(current_block_r1, mask_dataset)
        slope_deg = create_slope_matrix(terrain_dataset)
        
        remaining_height = mask_dataset.shape[0]
        remaining_width = mask_dataset.shape[1]
        
        write_win = Window.from_slices((slc[1], (slc[1] + remaining_height)), 
                                       ((slc[0], (slc[0] + remaining_width))))
        
        mask_export.write_band(1, mask_dataset.astype(rasterio.float32), window=write_win)
        slope_export.write_band(1, slope_deg.astype(rasterio.float32), window=write_win)

def write_rasters(in_raster1, in_raster2, kwargs, in_r1_nodata, in_r2_nodata):
    
    with rasterio.open(
        "mask.tiff",
        "w",
        **kwargs) as mask_export:
            
        with rasterio.open(
                "slopes.tiff", 
                "w",
                **kwargs) as slope_export:
            
            process_by_blocks(in_raster1, in_raster2, mask_export, slope_export, 
                              kwargs['width'], kwargs['height'], in_r1_nodata, in_r2_nodata)

def matrix_size(matrix):
    height = matrix.shape[0]
    width = matrix.shape[1]
    return height, width

def create_masking_matrix(dmp_matrix, dmt_matrix, dmp_nodata_val, dmt_nodata_val):
    THRESHOLD = 1
    masking_matrix = np.where((abs(dmp_matrix-dmt_matrix) < THRESHOLD)
                              & (dmt_matrix != dmt_nodata_val) 
                              & (dmp_matrix != dmp_nodata_val), 1, np.nan)
    return masking_matrix

def extract_terrain(dmp_matrix, masking_matrix):
    extraction = np.where(masking_matrix == 1, dmp_matrix, np.nan)
    return extraction

def create_slope_matrix (terrain_matrix):
    px, py = np.gradient(terrain_matrix, 1)
    slope = np.sqrt(px ** 2 + py ** 2)
    slope_deg = np.degrees(np.arctan(slope))
    return slope_deg
   
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
            
            
            kwargs = dmp.meta
                
            dmp_intersection, dmt_intersection, transform = get_raster_intersections(dmp, dmt)
            intersection_height, intersection_width = matrix_size(dmp_intersection)
            
            kwargs.update(driver="GTiff", dtype=rasterio.float32, compress='lzw',
                          height = intersection_height, width = intersection_width,
                          transform = transform)
            
            write_rasters(dmp_intersection, dmt_intersection, kwargs, dmp.nodata, dmt.nodata)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False, description = 'Skript, ktery ze vstupniho DMT a DMP\
        vytvori rastery sklonu nezastavenych ploch a nalezeny nezastavene plochy.')
    parser.add_argument('--terrain', action = "store", dest = "terraininput", required = True, help = 'Cesta k DMT ve formatu .tif.')
    parser.add_argument('--surface', action = "store", dest = "surfaceinput", required = True, help = 'Cesta k DMP ve formatu .tif.')
    args = parser.parse_args()
    proceed(args.surfaceinput, args.terraininput)