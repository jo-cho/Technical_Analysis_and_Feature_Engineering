# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 23:18:11 2021

@author: JHCho
"""
import pandas as pd
import numpy as np
import ta
from statsmodels.tsa.stattools import adfuller

def ohlcv(df):
    df.columns = [i.lower() for i in df.columns]
    close = pd.to_numeric(df.close)
    open = pd.to_numeric(df.open)
    high = pd.to_numeric(df.high)
    low = pd.to_numeric(df.low)
    volume = pd.to_numeric(df.volume)
    df_ohlcv = pd.DataFrame([open,high,low,close,volume]).T
    return df_ohlcv

def ta_stationary_test(df_):
    df = ohlcv(df_)
    all_ta = ta.add_all_ta_features(df,'open','high','low','close','volume')
    result_df = pd.DataFrame()
    for i in all_ta.columns:
        adf = adfuller(all_ta[i].dropna(),autolag='AIC')
        result_df['{}'.format(i)] = pd.Series(adf[0:2],index=['Test Statistic','p_value'])
    return result_df.T

def get_all_stationary_ta(df_, threshold=0.05):
    df = ohlcv(df_)
    all_ta = ta.add_all_ta_features(df,'open','high','low','close','volume')
    result_df = pd.DataFrame()
    for i in all_ta.columns:
        adf = adfuller(all_ta[i].dropna(),autolag='AIC')
        result_df['{}'.format(i)] = pd.Series(adf[0:2],index=['Test Statistic','p_value'])
    result = result_df.T
    all_ta = all_ta[result.loc[result.p_value<threshold].index]
    return all_ta

def log_diff(df, windows):
    mkt = df.copy()
    for i in windows:
        mkt = mkt.join(df.volume.pct_change(i).rename('vol_logdiff_{}'.format(i)))
        mkt = mkt.join(df.close.pct_change(i).rename('close_logdiff_{}'.format(i)))
    mkt = mkt.drop(columns=df.columns)
    return mkt

def mom_std(df, windows_mom, windows_std):
    mkt = df.copy()

    for i in windows_mom:
        mkt = mkt.join(df.volume.diff(i).rename('vol_mom_{}'.format(i)))
        mkt = mkt.join(df.close.diff(i).rename('mom_{}'.format(i)))

    for i in windows_std:
        mkt = mkt.join(df.close.rolling(i).std().rename('std_{}'.format(i)))
        mkt = mkt.join(df.volume.rolling(i).std().rename('vol_std_{}'.format(i)))
    mkt = mkt.drop(columns=df.columns)
    return mkt


#----------------------------------------- TA


from ta.momentum import (
    AwesomeOscillatorIndicator,
    KAMAIndicator,
    PercentagePriceOscillator,
    PercentageVolumeOscillator,
    ROCIndicator,
    RSIIndicator,
    StochasticOscillator,
    StochRSIIndicator,
    TSIIndicator,
    UltimateOscillator,
    WilliamsRIndicator,
)
from ta.others import (
    CumulativeReturnIndicator,
    DailyLogReturnIndicator,
    DailyReturnIndicator,
)
from ta.trend import (
    MACD,
    ADXIndicator,
    AroonIndicator,
    CCIIndicator,
    DPOIndicator,
    EMAIndicator,
    IchimokuIndicator,
    KSTIndicator,
    MassIndex,
    PSARIndicator,
    SMAIndicator,
    STCIndicator,
    TRIXIndicator,
    VortexIndicator,
)
from ta.volatility import (
    AverageTrueRange,
    BollingerBands,
    DonchianChannel,
    KeltnerChannel,
    UlcerIndex,
)
from ta.volume import (
    AccDistIndexIndicator,
    ChaikinMoneyFlowIndicator,
    EaseOfMovementIndicator,
    ForceIndexIndicator,
    MFIIndicator,
    NegativeVolumeIndexIndicator,
    OnBalanceVolumeIndicator,
    VolumePriceTrendIndicator,
    VolumeWeightedAveragePrice,
)


def get_stationary_ta_window_0(
    df_: pd.DataFrame,
    fillna: bool = False,
    mt = 1
) -> pd.DataFrame:
    """Add volume technical analysis features to dataframe.
    Args:
        df (pandas.core.frame.DataFrame): including ohlcv
    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
        
    mt: int: multiplier for windows (i set windows 10, 15 or 20)
    """
    df = ohlcv(df_)
    open='open';high='high';low='low';close='close';volume='volume'
    
    # Accumulation Distribution Index
    df["volume_adi"] = AccDistIndexIndicator(
        high=df[high], low=df[low], close=df[close], volume=df[volume], fillna=fillna
    ).acc_dist_index()

    # Chaikin Money Flow
    df["volume_cmf_{}".format(20*mt)] = ChaikinMoneyFlowIndicator(
        high=df[high], low=df[low], close=df[close], volume=df[volume], window=20*mt,
        fillna=fillna
    ).chaikin_money_flow()

    # Force Index
    df["volume_fi_{}".format(15*mt)] = ForceIndexIndicator(
        close=df[close], volume=df[volume], window=15*mt, fillna=fillna
    ).force_index()

    # Money Flow Indicator
    df["volume_mfi_{}".format(15*mt)] = MFIIndicator(
        high=df[high],
        low=df[low],
        close=df[close],
        volume=df[volume],
        window=15*mt,
        fillna=fillna,
    ).money_flow_index()

    # Ease of Movement
    indicator_eom = EaseOfMovementIndicator(
        high=df[high], low=df[low], volume=df[volume], window=15*mt, fillna=fillna
    )
    df["volume_em"] = indicator_eom.ease_of_movement()
    df["volume_sma_em_{}".format(15*mt)] = indicator_eom.sma_ease_of_movement()

    # Volume Price Trend
    df["volume_vpt"] = VolumePriceTrendIndicator(
        close=df[close], volume=df[volume], fillna=fillna
    ).volume_price_trend()


    # Average True Range
    df["volatility_atr_{}".format(10*mt)] = AverageTrueRange(
        close=df[close], high=df[high], low=df[low], window=10*mt, fillna=fillna
    ).average_true_range()

    # Bollinger Bands
    indicator_bb = BollingerBands(
        close=df[close], window=20*mt, window_dev=2, fillna=fillna
    )

    df["volatility_bbp_{}".format(20*mt)] = indicator_bb.bollinger_pband()


    # Keltner Channel
    indicator_kc = KeltnerChannel(
        close=df[close], high=df[high], low=df[low], window=10*mt, fillna=fillna
    )

    df["volatility_kcp_{}".format(10*mt)] = indicator_kc.keltner_channel_pband()


    # Donchian Channel
    indicator_dc = DonchianChannel(
        high=df[high], low=df[low], close=df[close], window=20*mt, offset=0, fillna=fillna
    )
    df["volatility_dcp_{}".format(20*mt)] = indicator_dc.donchian_channel_pband()

    # Ulcer Index
    df["volatility_ui_{}".format(15*mt)] = UlcerIndex(
        close=df[close], window=15*mt, fillna=fillna
    ).ulcer_index()
    
    # MACD
    indicator_macd = MACD(
        close=df[close], window_slow=25*mt, window_fast=10*mt, window_sign=9, fillna=fillna
    )
    df["trend_macd_{}_{}_{}".format(25*mt,10*mt,9)] = indicator_macd.macd()
    df["trend_macd_signal_{}_{}_{}".format(25*mt,10*mt,9)] = indicator_macd.macd_signal()
    df["trend_macd_diff_{}_{}_{}".format(25*mt,10*mt,9)] = indicator_macd.macd_diff()

    # Average Directional Movement Index (ADX)
    indicator_adx = ADXIndicator(
        high=df[high], low=df[low], close=df[close], window=15*mt, fillna=fillna
    )
    df["trend_adx_{}".format(15*mt)] = indicator_adx.adx()

    # Vortex Indicator
    indicator_vortex = VortexIndicator(
        high=df[high], low=df[low], close=df[close], window=15*mt, fillna=fillna
    )
    df["trend_vortex_ind_diff_{}".format(15*mt)] = indicator_vortex.vortex_indicator_diff()

    # TRIX Indicator
    df["trend_trix_{}".format(15*mt)] = TRIXIndicator(
        close=df[close], window=15*mt, fillna=fillna
    ).trix()

    # Mass Index
    df["trend_mass_index_{}_{}".format(10*mt,25*mt)] = MassIndex(
        high=df[high], low=df[low], window_fast=10*mt, window_slow=25*mt, fillna=fillna
    ).mass_index()

    # CCI Indicator
    df["trend_cci_{}".format(20*mt)] = CCIIndicator(
        high=df[high],
        low=df[low],
        close=df[close],
        window=20*mt,
        constant=0.015,
        fillna=fillna,
    ).cci()

    # DPO Indicator
    df["trend_dpo_{}".format(20*mt)] = DPOIndicator(
        close=df[close], window=20*mt, fillna=fillna
    ).dpo()


    # Aroon Indicator
    indicator_aroon = AroonIndicator(close=df[close], window=20, fillna=fillna)
    df["trend_aroon_ind_{}".format(20*mt)] = indicator_aroon.aroon_indicator()


    
    # Relative Strength Index (RSI)
    df["momentum_rsi_{}".format(15*mt)] = RSIIndicator(
        close=df[close], window=15*mt, fillna=fillna
    ).rsi()


    # TSI Indicator
    df["momentum_tsi_{}_{}".format(25*mt,15*mt)] = TSIIndicator(
        close=df[close], window_slow=25*mt, window_fast=15*mt, fillna=fillna
    ).tsi()


    # Stoch Indicator
    indicator_so = StochasticOscillator(
        high=df[high],
        low=df[low],
        close=df[close],
        window=15*mt,
        smooth_window=3,
        fillna=fillna,
    )
    df["momentum_stoch_{}".format(15*mt)] = indicator_so.stoch()
    df["momentum_stoch_signal_{}".format(15*mt)] = indicator_so.stoch_signal()

    # Williams R Indicator
    df["momentum_wr_{}".format(15*mt)] = WilliamsRIndicator(
        high=df[high], low=df[low], close=df[close], lbp=15*mt, fillna=fillna
    ).williams_r()

    # Awesome Oscillator
    df["momentum_ao_{}_{}".format(5*mt,35*mt)] = AwesomeOscillatorIndicator(
        high=df[high], low=df[low], window1=5*mt, window2=35*mt, fillna=fillna
    ).awesome_oscillator()

    # Rate Of Change
    df["momentum_roc_{}".format(10*mt)] = ROCIndicator(
        close=df[close], window=10*mt, fillna=fillna
    ).roc()

    # Percentage Price Oscillator
    indicator_ppo = PercentagePriceOscillator(
        close=df[close], window_slow=25*mt, window_fast=10*mt, window_sign=9, fillna=fillna
    )
    df["momentum_ppo_{}_{}_{}".format(25*mt,10*mt,9)] = indicator_ppo.ppo()
    df["momentum_ppo_signal_{}_{}_{}".format(25*mt,10*mt,9)] = indicator_ppo.ppo_signal()
    df["momentum_ppo_hist_{}_{}_{}".format(25*mt,10*mt,9)] = indicator_ppo.ppo_hist()
    
    df = df.drop(columns=['open','high','low','close','volume'])
    return df

def get_stationary_ta_windows(df,mts):
    TA = []
    for mt in mts:
        TA.append(get_stationary_ta_window_0(df,mt=mt))
    TA = pd.concat(TA,axis=1)
    TA = TA.loc[:,~TA.columns.duplicated()]
    TA = TA.reindex(sorted(TA.columns),axis=1)
    return TA

def get_my_ta(
    df_: pd.DataFrame,
    fillna: bool = False,
    window = 15
) -> pd.DataFrame:
    """Add volume technical analysis features to dataframe.
    Args:
        df (pandas.core.frame.DataFrame): including ohlcv
    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
        
    mt: int: multiplier for windows (i set windows 10, 15 or 20)
    """
    df = ohlcv(df_)
    open='open';high='high';low='low';close='close';volume='volume'

    # Chaikin Money Flow
    df["volume_cmf_{}".format(window)] = ChaikinMoneyFlowIndicator(
        high=df[high], low=df[low], close=df[close], volume=df[volume], window=window,
        fillna=fillna
    ).chaikin_money_flow()

    # Ulcer Index
    df["volatility_ui_{}".format(window)] = UlcerIndex(
        close=df[close], window=window, fillna=fillna
    ).ulcer_index()
    
    # DPO Indicator
    df["trend_dpo_{}".format(window)] = DPOIndicator(
        close=df[close], window=window, fillna=fillna
    ).dpo()

    # Relative Strength Index (RSI)
    df["momentum_rsi_{}".format(window)] = RSIIndicator(
        close=df[close], window=window, fillna=fillna
    ).rsi()
    
    df = df.drop(columns=['open','high','low','close','volume'])
    return df
    
def get_my_ta_windows(df,windows):
    TA = []
    for mt in windows:
        TA.append(get_my_ta(df,window=mt))
    TA = pd.concat(TA,axis=1)
    TA = TA.loc[:,~TA.columns.duplicated()]
    TA = TA.reindex(sorted(TA.columns),axis=1)
    return TA

#------------------------------------ RSI
def get_rsi(price,windows):
    df = pd.DataFrame()
    for w in windows:
        df['rsi_{}'.format(w)] = ta.momentum.rsi(price,w)
    return df

def get_rsi_decision(close,windows):
    df = pd.DataFrame()
    for w in windows:
        rsi = ta.momentum.rsi(close,window=w)
        # overbought
        d1 = (rsi > 70)
        # oversold
        u1 = (rsi < 30)
        # up trend
        u2 = ~(d1)&~(u1)&(rsi.pct_change() >= 0)
        #down trend
        d2 = ~(d1)&~(u1)& (rsi.pct_change() <0)

        rsi_ind = rsi.dropna().copy()
        rsi_ind.loc[d1] = -1;rsi_ind.loc[d2] = -1
        rsi_ind.loc[u1] = 1;rsi_ind.loc[u2] = 1
        df['rsid_{}'.format(w)] = rsi_ind[1:]
    return df
#-----------------------------------------

