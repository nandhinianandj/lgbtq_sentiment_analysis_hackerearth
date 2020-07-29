require(NNS)
require(knitr)
require(rgl)
require(data.table)
require(dtw)

housing_table = read.table('/home/data/california-housing-prices.csv', sep=',')
nns = NNS.ARMA(housing_table, h = 44, training.set = 100, method = "lin", plot = TRUE, seasonal.factor = 12, seasonal.plot = FALSE, ncores = 1)

sqrt(mean((nns - tail(housing_table, 44)) ^ 2))

nns = NNS.ARMA(housing_table, h = 44, training.set = 100, method = "nonlin", plot = FALSE, seasonal.factor = 12, seasonal.plot = FALSE, ncores = 1)

sqrt(mean((nns - tail(housing_table, 44)) ^ 2))

seas = t(sapply(1 : 25, function(i) c(i, sqrt( mean( (NNS.ARMA(housing_table, h = 44, training.set = 100, method = "lin", seasonal.factor = i, plot=FALSE, ncores = 1) - tail(housing_table, 44)) ^ 2) ) ) ) )

colnames(seas) = c("Period", "RMSE")

a = seas[which.min(seas[ , 2]), 1]

nns = NNS.ARMA(housing_table, h = 44, training.set = 100, method = "nonlin", seasonal.factor = a, plot = TRUE, seasonal.plot = FALSE, ncores = 1)

sqrt(mean((nns - tail(housing_table, 44)) ^ 2))

NNS.seas(housing_table, modulo = 12, plot = FALSE)

nns.optimal = NNS.ARMA.optim(housing_table[1:100],
                             training.set = 88,
                             seasonal.factor = seq(12, 24, 6),
                             obj.fn = expression( sqrt(mean((predicted - actual)^2)) ), ncores = 1)


nns = NNS.ARMA(housing_table, training.set = 100, h = 44, seasonal.factor = nns.optimal$periods, weights = nns.optimal$weights,
	       method = nns.optimal$method, plot = TRUE, seasonal.plot = FALSE, ncores = 1)

sqrt(mean((nns - tail(housing_table, 44)) ^ 2))

sqrt(mean((nns+nns.optimal$bias.shift - tail(housing_table, 44)) ^ 2))

nns <- pmax(0, nns+nns.optimal$bias.shift)
sqrt(mean((nns - tail(housing_table, 44)) ^ 2))

NNS.ARMA(housing_table, h = 50, seasonal.factor = nns.optimal$periods, method  = nns.optimal$method, weights = nns.optimal$weights, plot = TRUE, seasonal.plot = FALSE, ncores = 1) + nns.optimal$bias.shift




