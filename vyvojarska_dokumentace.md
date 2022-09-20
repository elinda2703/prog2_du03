# Vývojářská dokumentace

## Parser a vstupní soubory

Program je sestaven tak, aby se spouštěl z terminálu. Pomocí knihovny `argparse` byl vytvořen parser, který umožňuje zadat v terminálu parametry
`--terrain <cesta_k_DMT>` a `--surface <cesta_k_DMP>`. Oba parametry jsou povinné. Tímto způsobem může uživatel specifikovat cestu k vstupním rastrům, aniž by byly v stejné složce, jako je skript `ukol3.py`.

Protože program pracuje s informacemi, které jsou v podporované formě obsaženy jen v rastrových souborech, oba vstupní soubory musí být rastrového formátu. Jedná se například o porovnání souřadnicových systémů nebo extrakci metadat.

Pro zpracování vstupních souborů jsou využívany knihovny `rasterio`, `shapely` a `numpy`, které je potřeba před spuštěním programu nainstalovat.

## Tolerance pro identifikaci shodného povrchu

Před samotným spuštěním programu je vhodné zkontrolovat a případně změnit hodnotu konstanty `THRESHOLD`, která se nachází ve funkci `proceed()`. Tato konstanta představuje toleranci, do jakého rozdílu
je terén na stejné ploše v DMT i v DMP považován za shodný. Udáva se v metrech. Přednastavená hodnota je `1` metr. Pokud chce uživatel dosáhnout přesnejších výsledků, je potřeba tuto hodnotu znížit.

## Použity funkce

Podrobnější popisy výpočtů a operací jsou popsány v kódu.

### get_raster_intersections(raster1, raster2)

Funkce najde průniky obou vstupních rastrů. Vrací matice průniků pro DMT i pro DMP a jejich transformaci, která je shodná.

### matrix_size(matrix)

Funkce vrátí výšku a šířku vstupní matice.

Z výše uvedených funkcí se vytvoří metadata pro výstupní soubory.

### write_rasters(in_raster1, in_raster2, kwargs, in_r1_nodata, in_r2_nodata, threshold)

Funkce vytvoří objekty typu `io.DatasetWriter` (z knihovny `rasterio`), které slouží pro zápis rastrů.
Následně zavolá funkci `process_by_blocks()`, která provede samotný zápis.

### process_by_blocks(r1_matrix, r2_matrix, mask_export, slope_export, intersection_width, intersection_height, r1_nodata, r2_nodata, threshold)

Protože vstupní rastry můžou být velké, je potřebné je zpracovat po částech, aniž by došla paměť. Tyto části reprezentují tzv. bloky. Ve funkci se vytvoří tolik bloků, aby pokryly celý
průnik rastrů, a následně se blok po bloku postupně převedou výpočty pomocí funkcí `create_masking_matrix`, `extract_terrain` a `create_slope_matrix`. Vypočteny hodnoty se v blocích postupně zapisují do výstupních souborů.
V programu je velikost bloků nastavena na `128x128` pixelů, tuto hodnotu je možné měnit.

### create_masking_matrix(dmp_matrix, dmt_matrix, dmp_nodata_val, dmt_nodata_val, threshold)

Funkce vytvoří matici s hodnotami `numpy.nan` a `1`. Hodnota `1` představuje nezastavěné plochy určeny konstantou `THRESHOLD`, hodnota `numpy.nan` se dosadí za zanedbané plochy. Dosazení této hodnoty
je nezbytné, protože se s ní dále pracuje ve funkci `create_slope_matrix()`.

### extract_terrain(dmp_matrix, masking_matrix)

Funkce extrahuje pomocí předem vytvořené masky ve funkci `create_masking_matrix` nezastavěný povrch z DMT.

### create_slope_matrix (terrain_matrix)

Funkce vytvoří matici sklonu terénu, který byl již extrahován ve funkci `extract_terrain`. Výsledná matice obsahuje hodnoty sklonitosti v stupních.

# Výstup

Po spuštění programu v terminálu jsou vytvořeny soubory `mask.tiff` a `slopes.tiff` ve formátu GeoTIFF.