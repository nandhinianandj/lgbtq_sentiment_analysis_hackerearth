library(PSF)
df = read.csv('/home/data/california-housing-prices.csv')
future_preds = 10
#plot(df, predictions=c(), cycle=24)
opt_w = optimum_w(df, future_preds)
opt_k = optimum_k(df, future_preds)
pred_for_w(df,opt_w, opt_k,future_preds)

p<-psf(nottem)
pred<-predict(p, n.ahead=12)
