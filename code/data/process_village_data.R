library(data.table)
library(sf)
library(stringr)

PROJECT_DIR <- '~/Projects/aii-design'
MAP_DATA_DIR <- file.path(PROJECT_DIR,'data','raw','Thailand Geodata')
LOSS_DATA_DIR <- file.path(PROJECT_DIR,'data','raw','Thailand Loss')

load_map_data <- function(){
    fpath <- file.path(MAP_DATA_DIR,'Adm3.shp')
    df <- read_sf(fpath)

}

load_loss_data <- function(){
    fpath <- file.path(LOSS_DATA_DIR,'Tambon_loss_data.csv')
    df <- fread(fpath)
    df[, c('Province','Amphur','TCode') := tstrsplit(Tambon,'-')]
    df[nchar(Amphur) == 1, Amphur := paste0('0',Amphur)]
    df[nchar(TCode) == 1, TCode := paste0('0',TCode)]
    df[, ADM3_PCODE := paste0('TH',Province, Amphur, TCode)]
    return(df)
}

sdf <- load_map_data()
df <- load_loss_data()
tdf <- df[, .N, by=ADM3_PCODE]
tst <- merge(sdf, tdf, by='ADM3_PCODE')
st_write(tst,'Thailand.shp')