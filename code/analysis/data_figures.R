library(data.table)
library(ggplot2)

PROJECT_DIR <- '~/Projects/aii-design'

yield_data_path <- file.path(PROJECT_DIR,'data','processed','midwest_yield_data.csv')
df <- fread(yield_data_path)
tst <- df[,.(Yield=mean(Value)),by=.(State,Year)]
tst <- df[,.(Yield=mean(`TS Value`)),by=.(State,Year)]

ggplot(tst, aes(x=Year,y=Yield))+geom_line()+facet_wrap(vars(State))+theme_minimal()

corr_plot <- function(){
    yield_data_path <- file.path(PROJECT_DIR,'data','processed','midwest_yield_data.csv')
    df <- fread(yield_data_path)
    tst <- df[,.(Yield=mean(Value)),by=.(State,Year)]
    tst <- df[,.(Yield=mean(`TS Value`)),by=.(State,Year)]

    
}