#install.packages("remotes")
#remotes::install_github("RamiKrispin/forecastLM")

library(forecastLM)

data("ny_gas")

class(ny_gas)
## [1] "tbl_ts"     "tbl_df"     "tbl"        "data.frame"
head(ny_gas)

library(TSstudio)

ts_plot(ny_gas,
        title = "The New York Natural Gas Residential Monthly Consumption",
        Ytitle = "Million Cubic Feet",
        Xtitle = "Source: US Energy Information Administration (Jan 2020)")

md1 <- trainLM(input = ny_gas,
               y = "y",
               trend = list(linear = TRUE),
               seasonal = "month")

summary(md1)
summary(md1$model)

plot_res(md1)

md2 <- trainLM(input = ny_gas,
               y = "y",
               trend = list(linear = TRUE),
               seasonal = "month",
               lags = c(1,12))

summary(md2$model)

head(md2$series, 13)

plot_res(md2)

md3 <- trainLM(input = ny_gas,
               y = "y",
               trend = list(linear = TRUE),
               seasonal = "month",
               lags = c(1, 6, 12),
               step = TRUE)

summary(md3$model)

events1 <- list(outlier = c(as.Date("2015-01-01"), as.Date("2015-02-01"), as.Date("2018-01-01"), as.Date("2019-01-01")))

md4 <- trainLM(input = ny_gas,
               y = "y",
               trend = list(linear = TRUE),
               seasonal = "month",
               lags = c(1, 12),
               events = events1,
               step = TRUE)

summary(md4$model)
plot_res(md4)

fc4 <- forecastLM(md4, h = 60)
plot_fc(fc4)
plot_fc(fc4, theme = "darkPink")



