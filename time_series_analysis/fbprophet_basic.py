# -*- coding: utf-8 -*-
# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.#

#* File Name : fbprophet_basic.py
#
#* Purpose :
#
#* Creation Date : 17-01-2020
#
#* Last Modified : Friday 17 January 2020 08:04:33 PM IST
#
#* Created By :

#_._._._._._._._._._._._._._._._._._._._._.#

# Python
import pandas as pd
from fbprophet import Prophet


def fbprophet_preds(df, periods=365, growth='logistic'):
    m = Prophet(growth=growth)
    if growth=='logistic':
        df['cap'] = 500001.00
        df['floor'] = 10000.00
    m.fit(df)

    future = m.make_future_dataframe(periods=periods)
    if growth=='logistic':
        future['cap'] = 500001.00
        future['floor'] = 14999.00
    forecast = m.predict(future)
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

    return forecast

#fig1 = m.plot(forecast)
#fig2 = m.plot_components(forecast)
#
#fig = plot_plotly(m, forecast)  # This returns a plotly Figure
#py.iplot(fig)

def main():
    df = pd.read_csv('/home/data/california-housing-prices.csv')
    df.rename(columns={'housing_median_age': 'ds'}, inplace=True)
    df.rename(columns={'median_house_value': 'y'}, inplace=True)
    import pdb; pdb.set_trace()
    print(fbprophet_preds(df, periods=1, growth='logistic'))
if __name__ == '__main__':
    main()
