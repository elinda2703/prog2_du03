# Uživatelská dokumentace

## O programu

Program nalezne nezastavěné plochy ze vstupních rastrů DMT (Digitální model terénu) a DMP (Digitální model povrchu) ve formátu GeoTIFF. Výstupem jsou dva soubory: `mask.tiff`, reprezentující plochy bez zástavby
a vegetace, a `slopes.tiff`, obsahující vypočtený sklon terénu na místech identifikovaných maskou.

## Použití

Doporučujeme vytvořit si novou složku, do které umístníte soubor `ukol3.py`. V této složce budou zároveň vytvořeny oba výstupní soubory.
Před spuštěním zkontrolujte, jestli vám vyhovuje přednastavená hodnota tolerance (1 metr, proměnná `THRESHOLD`), pomocí které se zjišťuje shodnost povrchu obou vstupních rastrů. Čím vyšší je tato hodnota,
tím větší rozdíl nadmořských výšek mezi DMP a DMT se na stejné ploše připustí, co snižuje přesnost identifikace skutečných nezastavěných ploch.

Program se spouští z terminálu příkazem `python ukol3.py --terrain <cesta_k_DMT> --surface <cesta_k_DMP>`. Oba parametry jsou nezbytné pro chod programu.
Po spuštění program zkontroluje, zda-li mají vstupní rastry shodný souřadnicový systém – pokud tomu tak není, program na to upozorní a ukončí se, protože nebude schopen provést další operace.
Následně se na průniku vstupních rastrů DMP a DMT vypočtou a vytvoří oba výstupní soubory. Sklon terénu v souboru `slopes.tiff` je vypočten z DMT, co může uživatel v případě potřeby změnit.
