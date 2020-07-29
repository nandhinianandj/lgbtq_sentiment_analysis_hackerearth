# -*- coding: utf-8 -*-
# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.#

#* File Name :
#
#* Purpose :
#
#* Creation Date : 04-07-2020
#
#* Last Modified : Saturday 04 July 2020 10:30:59 PM IST
#
#* Created By :

#_._._._._._._._._._._._._._._._._._._._._.#


def kaplan_meier_filter(dataframe, timeCol, targetCol):
    from lifelines import KaplanMeierFitter

    durations = [ 11, 74, 71, 76, 28, 92, 89, 48, 90, 39, 63, 36, 54, 64, 34, 73, 94, 37, 56, 76, ]
    event_observed = [ True, True, False, True, True, True, True, False, False, True, True, True, True, True, True, True, False, True, False, True, ] 
    kmf = KaplanMeierFitter()
    kmf.fit(dataframe[timeCol], dataframe[targetCol])
    kmf.plot()


def survival_analyze(
    dataframe, lifetime_col, dead_col, strata_cols, covariate_col=None
):
    # Based on notebook here. https://github.com/CamDavidsonPilon/lifelines/tree/master/examples
    import pandas as pd
    from matplotlib import pyplot as plt
    from lifelines import CoxPHFitter

    cph = CoxPHFitter().fit(dataframe, lifetime_col, dead_col, strata=strata_cols)
    cph.plot(ax=ax[1])
    if covariate_col:
        cph.plot_covariate_groups(covariate_col, values=[0, 1])
    pass

def fbprophet_forecast(dataframe,):
    assert "ds" in dataframe.columns, "column ds needed to be the time column"
    from fbprophet import Prophet

    m = Prophet()
    m.fit(dataframe)
    future = m.make_future_dataframe(periods=365)
    forecast = m.predict(future)
    print(forecast)
    m.plot(forecast)


def fbprophet_multiplicative_seasonality(dataframe, timeCol):
    assert "ds" in dataframe.columns, "column ds needed to be the time column"
    m = Prophet(seasonality_mode="multiplicative")
    m.fit(dataframe)
    future = m.make_future_dataframe(50, freq="MS")
    forecast = m.predict(future)
    print(forecast)
    m.plot(forecast)
    m.plot_components(forecast)


def fbprophet_sub_daily_data(dataframe):
    assert "ds" in dataframe.columns, "column ds needed to be the time column"
    m = Prophet(changepoint_prior_scale=0.01).fit(dataframe)
    future = m.make_future_dataframe(periods=300, freq="H")
    fcst = m.predict(future)
    m.plot(fcst)
    m.plot_components(forecast)


def fbprophet_w_uncertainty(dataframe, estimate_type="mcmc"):
    assert "ds" in dataframe.columns, "column ds needed to be the time column"
    if estimate_type == "mcmc":
        m = Prophet(mcmc_samples=300)
    else:
        # MAP estimate
        m = Prophet(interval_width=0.95)

    forecast = m.fit(dataframe).predict(future)
    m.plot(fcst)
    m.plot_components(forecast)


def fbprophet_changepoint_detect(dataframe):
    assert "ds" in dataframe.columns, "column ds needed to be the time column"
    m = Prophet()
    m.fit(dataframe)
    future = m.make_future_dataframe(periods=366)
    forecast = m.predict(future)
    fig = m.plot(forecast)
    for cp in m.changepoints:
        plt.axvline(cp, c="gray", ls="--", lw=2)


def tsfresh_extract_features(timeSeries, idCol, timeCol):
    from tsfresh import extract_relevant_features
    from tsfresh import select_features
    from tsfresh.utilities.dataframe_functions import impute

    extracted_features = extract_relevant_features(
        timeSeries, column_id=idCol, column_sort=timeCol
    )

    impute(extracted_features)
    features_filtered = select_features(extracted_features, y)
    return features_filtered


def tsfresh_sklearn_transform(dataframe):
    assert "id" in dataframe.columns, "dataframe needs id column"
    assert "time" in dataframe.columns, "Need time column in dataframe"
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    from tsfresh.examples import load_robot_execution_failures
    from tsfresh.transformers import RelevantFeatureAugmenter

    pipeline = Pipeline(
        [
            ("augmenter", RelevantFeatureAugmenter(column_id="id", column_sort="time")),
            ("classifier", RandomForestClassifier()),
        ]
    )

    df_ts, y = load_robot_execution_failures()
    X = pd.DataFrame(index=y.index)

    pipeline.set_params(augmenter__timeseries_container=df_ts)
    pipeline.fit(X, y)
