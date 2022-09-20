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
    '''
    Funkce ziskava prunik vstupnich rastru (v nasem pripade DMP a DMT).

        Parametry:
            raster1 (objekt typu io.DatasetReader): Vstupni DMP raster.
            raster2 (objekt typu io.DatasetReader): Vstupni DMT raster.

        Vraci:
            r1win (array): Matice hodnot pixelu spolecniho pruniku pro prvni raster.
            r2win (array): Matice hodnot pixelu spolocneho pruniku pro druhy raster.
            transform (objekt typu Affine): Transformace spolecneho pruniku.
    '''
    # Vytvoreni bounding boxu ohranicujicich oba rastry,
    # datovy typ polygon z knihovny shapely
    bb_raster1 = box(raster1.bounds[0], raster1.bounds[1], raster1.bounds[2], raster1.bounds[3])
    bb_raster2 = box(raster2.bounds[0], raster2.bounds[1], raster2.bounds[2], raster2.bounds[3])
    
    if bb_raster1.intersects(bb_raster2):
        pass
    
    else:
        sys.exit("Vstupni rastry se neprekryvaji.")
    
    # Ziskani krajnich souradnic obou rasteru
    xminR1, yminR1, xmaxR1, ymaxR1 = raster1.bounds
    xminR2, yminR2, xmaxR2, ymaxR2 = raster2.bounds
    
    # Vytvoreni pruniku, datovy typ polygon z knihovny shapely
    intersection = bb_raster1.intersection(bb_raster2)
    
    # Vytvoreni transformace pruniku
    transform = Affine(raster1.res[0], 0.0, intersection.bounds[0], 0.0, -raster1.res[1], intersection.bounds[3])
    
    # Ziskani krajnich souradnic pruniku s offsetem poloviny 
    # vysky a sirky pixelu daneho souradnicoveho systemu
    p1Y = intersection.bounds[3] - raster1.res[1]/2
    p1X = intersection.bounds[0] + raster1.res[0]/2
    p2Y = intersection.bounds[1] + raster1.res[1]/2
    p2X = intersection.bounds[2] - raster1.res[0]/2
    
    # Ziskani prvniho radku a sloupce pruniku
    # z obou puvodnich rasteru
    row1R1 = int((ymaxR1 - p1Y)/raster1.res[1])
    row1R2 = int((ymaxR2 - p1Y)/raster2.res[1])
    col1R1 = int((p1X - xminR1)/raster1.res[0])
    col1R2 = int((p1X - xminR2)/raster2.res[0])

    # Ziskani posledniho radku a sloupce pruniku
    # z obou puvodnich rasteru
    row2R1 = int((ymaxR1 - p2Y)/raster1.res[1])
    row2R2 = int((ymaxR2 - p2Y)/raster2.res[1])
    col2R1 = int((p2X - xminR1)/raster1.res[0])
    col2R2 = int((p2X - xminR2)/raster2.res[0])
    
    # Vypocitani sirky a vysky pruniku v pixelech
    width1 = col2R1 - col1R1 + 1
    width2 = col2R2 - col1R2 + 1
    height1 = row2R1 - row1R1 + 1
    height2 = row2R2 - row1R2 + 1
    
    # Vytvoreni matic spolecneho pruniku z obou rasteru
    r1win = raster1.read(1, window=Window(col1R1, row1R1, width1, height1))
    r2win = raster2.read(1, window=Window(col1R2, row1R2, width2, height2))
    
    return r1win, r2win, transform

def process_by_blocks(r1_matrix, r2_matrix, mask_export, slope_export, 
                      intersection_width, intersection_height, r1_nodata, r2_nodata,
                      threshold):
    '''
    Funkce spracuje a provede vypocty na maticich prusecniku rastru po blocich, 
    aniz by dosla pamet. Spraocvane bloky zapise do vystupnich rastru.

        Parametry:
            r1_matrix (array): Matice hodnot pixelu spolecniho pruniku pro prvni raster.
            r2_matrix (array): Matice hodnot pixelu spolocneho pruniku pro druhy raster.
            mask_export (objekt typu io.DatasetWriter): Objekt knihovny rasterio, ktery umoznuje zapis do vystupniho souboru.
            slope_export (objekt typu io.DatasetWriter): Objekt knihovny rasterio, ktery umoznuje zapis do vystupniho souboru.
            intersection_width (int): Sirka pruniku rastru v pixelech.
            intersection_height (int): Vyska pruniku rastru v pixelech.
            r1_nodata (float): Hodnota NoData pixelu z prvniho rasteru, potrebna na vypocet vystupniho povrchu. 
            r2_nodata (float): Hodnota NoData pixelu z druheho rasteru, potrebna na vypocet vystupniho povrchu.
            threshold (int): Stanovena tolerance pro identifikaci shodneho povrchu. 
    '''
    
    # Stanoveni velikosti bloku pro zpracovani.
    # Defaultni hodnota je 128.
    block_size = 128
    
    # Vytvoreni seznamu obsahujiciho ctverice charakterizujici jeden blok.
    # Prvni dve hodnoty v ctverici charakterizuji prvni sloupec a radek v bloku,
    # treti a ctvrta hodnota jsou sirka a vyska bloku.
    # Bloky jsou vytvoreny tak, aby pokryvaly celou matici pruniku.
    slices = [(col_start, row_start, block_size, block_size) \
                for col_start in list(range(0, intersection_width, block_size)) \
                for row_start in list(range(0, intersection_height, block_size))
    ]
        
    # Iterace pro kazdy blok v seznamu
    for slc in slices:
        
        # Vyrez prave iterovaneho bloku z matic pruniku
        current_block_r1 = r1_matrix[(slc[1]):(slc[1] + block_size), slc[0]:(slc[0] + block_size)]
        current_block_r2 = r2_matrix[(slc[1]):(slc[1] + block_size), slc[0]:(slc[0] + block_size)]
        
        # Provedeni vypoctu pro prave iterovany blok
        mask_dataset = create_masking_matrix(current_block_r1, current_block_r2, 
                                             r1_nodata, r2_nodata, threshold)
        terrain_dataset = extract_terrain(current_block_r2, mask_dataset)
        slope_deg = create_slope_matrix(terrain_dataset)
        
        # Pojistka pro pripad, kdy hranice bloku muzou presahovat matici pruniku.
        # Kdyz takto neni mozne vytvorit okno pro zapis, velikost okna se prispusobi
        # zbyvajicim radkum a sloupcum z matice.
        remaining_height = mask_dataset.shape[0]
        remaining_width = mask_dataset.shape[1]
        
        # Vytvoreni zapisovaciho okna, ktereho velikost
        # se snazi drzet velikosti stanoveneho bloku,
        # v pripade potreby se zmensuje
        write_win = Window.from_slices((slc[1], (slc[1] + remaining_height)), 
                                       ((slc[0], (slc[0] + remaining_width))))
        
        # Zapis bloku do vystupnich rasteru.
        mask_export.write_band(1, mask_dataset.astype(rasterio.float32), window=write_win)
        slope_export.write_band(1, slope_deg.astype(rasterio.float32), window=write_win)

def write_rasters(in_raster1, in_raster2, kwargs, in_r1_nodata, in_r2_nodata, threshold):
    '''
    Funkce vytvori objekty slouzici pro zapis vystupnich rasteru.
    Nasledne zavola funkci, ktera vytvori vystupni rastry.

        Parametry:
            in_raster1 (array): Matice hodnot pixelu spolecniho pruniku pro prvni raster.
            in_raster2 (array): Matice hodnot pixelu spolocneho pruniku pro druhy raster.
            kwargs (dict): Slovnik obsahujici metadata o vystupech.
            in_r1_nodata (float): Hodnota NoData pixelu z prvniho rasteru, potrebna na vypocet vystupniho povrchu. 
            in_r2_nodata (float): Hodnota NoData pixelu z druheho rasteru, potrebna na vypocet vystupniho povrchu.
            threshold (int): Stanovena tolerance pro identifikaci shodneho povrchu. 
    '''
    
    # Vytvoreni objektu tridy DatasetWriter
    with rasterio.open(
        "mask.tiff",
        "w",
        **kwargs) as mask_export:
            
        with rasterio.open(
                "slopes.tiff", 
                "w",
                **kwargs) as slope_export:
            
            # Zavolani funkce pro vytvoreni vystupu
            process_by_blocks(in_raster1, in_raster2, mask_export, slope_export, 
                              kwargs['width'], kwargs['height'], in_r1_nodata, in_r2_nodata,
                              threshold)

def matrix_size(matrix):
    '''
    Funkce vraci vysku a sirku matice.

        Parametry:
            matrix (array): Vstupni matice.

        Vraci:
            height (int): Vyska matice.
            width (int): Sirka matice.
    '''
    height = matrix.shape[0]
    width = matrix.shape[1]
    return height, width

def create_masking_matrix(dmp_matrix, dmt_matrix, dmp_nodata_val, dmt_nodata_val, threshold):
    '''
    Funkce vytvori a vrati masku, ktera popisuje povrch.
    Hodnota 1 identifikuje nezastavene oblasti, kde se nadmorske vysky DMP a DMT pohybuji
    uvnitr stanovene tolerance. Hodnota numpy.nan identifikuje mista, kde je zastavba, stromy i jine prekazky.

        Parametry:
            dmp_matrix (array): Vstupni matice DMP.
            dmt_matrix (array): Vstupni matice DMT.
            dmp_nodata_val (float): Hodnota NoData pixelu z DMP.
            dmt_nodata_val (float): Hodnota NoData pixelu z DMT.
            threshold (int): Stanovena tolerance pro identifikaci shodneho povrchu.

        Vraci:
            masking_matrix (array): Matice o hodnotach 0 a 1 identifikujici nezastavene oblasti.
    '''
    
    # Kde je rozdil nadmorskych vysek pro dany pixel mensi ako stanovena tolerance
    # a zaroven se hodnota daneho pixelu nerovna hodnote pro NoData,
    # povrch se prohlasi za shodny a dosadi se hodnota 1.
    # Jinak se dosadi numpy.nan z knihovny NumPy, potrebna pro dalsi operace s touto matici.
    masking_matrix = np.where((abs(dmp_matrix-dmt_matrix) < threshold)
                              & (dmt_matrix != dmt_nodata_val) 
                              & (dmp_matrix != dmp_nodata_val), 1, np.nan)
    return masking_matrix

def extract_terrain(dmt_matrix, masking_matrix):
    '''
    Funkce extrahuje nezastaveny povrch z DMT (muze i z DMP) pomoci maskovaci matice.

        Parametry:
            dmp_matrix (array): Vstupni matice DMP.
            masking_matrix (array): Matice o hodnotach 0 a 1 identifikujici nezastavene oblasti.

        Vraci:
            extraction (array): Matice s nadmorskymi vyskami nezastavenych oblasti.
    '''
    extraction = np.where(masking_matrix == 1, dmt_matrix, np.nan)
    return extraction

def create_slope_matrix (terrain_matrix):
    '''
    Funkce vytvori a vrati matici sklonu terenu ze vstupni matice nadmorskych vysek.

        Parametry:
            terrain_matrix (array): Matice nadmorskych vysek.

        Vraci:
            slope_deg (array): Matice sklonu terenu v stupnich.
    '''
    
    # Vypocteni parcialnych derivaci (gradientu) v obou osach x, y
    px, py = np.gradient(terrain_matrix, 1)
    
    # Vypocteni sklonu v radianech
    slope = np.sqrt(px ** 2 + py ** 2)
    
    # Konverze radianu na stupne
    slope_deg = np.degrees(np.arctan(slope))
    return slope_deg
   
def proceed(surfaceinput, terraininput):
    '''
    Funkce vezme vstupni rastry DMP a DMT a na jich pruniku vytvori 2 nove rastry:
    masku identifikujici nezastavene oblasti a raster sklonitosti terenu extrahovaneho maskou.

        Parametry:
            surfaceinput (objekt typu io.DatasetReader): Vstupni raster DMP.
            terraininput (objekt typu io.DatasetReader): Vstupni raster DMT
    '''
    
    # Otevreni obou rastru pomoci knihovny rasterio

    with rasterio.open(surfaceinput) as dmp:
    
        with rasterio.open(terraininput) as dmt:
            
            try:
                # Souradnicove systemy se musi zhodovat kvuli velikosti pixelu a lokalizaci.
                # Kdyz se neshoduji, nelze provest dalsi vypocty.
                if dmp.crs == dmt.crs:
                    pass
                
                else:
                    sys.exit("Zvolene rastery nemaji zhodny souradnicovy system, nebo je v nich ruzne zadefinovany. Nelze provest dalsi vypocty.")
            
            except rasterio.errors.CRSError(dmp):
                sys.exit("Ve zvolenem surface rasteru neni definovany validny souradnicovy system.")
                
            except rasterio.errors.CRSError(dmt):
                sys.exit("Ve zvolenem terrain rasteru neni definovany validny souradnicovy system.")  
            
            except rasterio.error.RasterioIOError:
                sys.exit("Vstupni soubory nejsou rastroveho formatu.")
            
            except rasterio.errors.RasterioError():
                sys.exit("Neco se nepovedlo, program se ted ukonci.")
            
            # Kdyz jsou souradnicove systemy totozne, extrahuji se metadata jedneho z rastru
            # (v nasem pripade z DMP, pro nase ucely na tom ale nezalezi).
            kwargs = dmp.meta
            
            # Tolerance pro identifikaci shodneho povrchu v METRECH. Je mozne prispusobit.
            THRESHOLD = 1
            
            # Zavolani funkci pro nalezeni pruniku rastru a zjisteni jeho rozmeru
            dmp_intersection, dmt_intersection, transform = get_raster_intersections(dmp, dmt)
            intersection_height, intersection_width = matrix_size(dmp_intersection)
            
            # Na zaklade nalezeneho pruniku se upravi metadata, ktere budou
            # dosazeny do obou vystupu
            kwargs.update(driver="GTiff", dtype=rasterio.float32, compress='lzw',
                          height = intersection_height, width = intersection_width,
                          transform = transform)
            
            # Samotne vytvoreni vystupu
            write_rasters(dmp_intersection, dmt_intersection, kwargs, dmp.nodata, dmt.nodata, THRESHOLD)
            
'''
Parser umoznujici zadat vstupni rastry z prikazove radky
'''            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False, description = 'Skript, ktery ze vstupniho DMT a DMP\
        vytvori rastery sklonu nezastavenych ploch a nalezeny nezastavene plochy.')
    parser.add_argument('--terrain', action = "store", dest = "terraininput", required = True, help = 'Cesta k DMT ve formatu .tif.')
    parser.add_argument('--surface', action = "store", dest = "surfaceinput", required = True, help = 'Cesta k DMP ve formatu .tif.')
    args = parser.parse_args()
    proceed(args.surfaceinput, args.terraininput)
