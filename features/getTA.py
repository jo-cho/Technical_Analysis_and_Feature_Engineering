import pandas as pd 
import numpy as np 
from ta import momentum, trend, volatility, volume 
import ta


def get_ta(df, mt):
    """
    mt: multiplier for windows
    """
    ti = pd.DataFrame()
    close=pd.to_numeric(df.close);high=pd.to_numeric(df.high)
    open=pd.to_numeric(df.open);low=pd.to_numeric(df.low)
    volume=pd.to_numeric(df.volume)
    
    ti['rsi_{}'.format(14*mt)] = ta.momentum.rsi(close,14*mt)
    ti['stoch_diff_{}_3'.format(14*mt)] = (ta.momentum.stoch(high,low,close,14*mt)
                                           - ta.momentum.stoch_signal(high,low,close,14*mt))
    ti['macd_diff_{}_{}_9'.format(26*mt,12*mt)] = ta.trend.macd_diff(close,26*mt,12*mt,10)
    ti['dpo_{}'.format(20*mt)] = ta.trend.dpo(close,20*mt)
    ti['aroon_{}'.format(25*mt)] = ta.trend.AroonIndicator(close,25*mt).aroon_indicator()
    ti['mfv'] = (volume* ((close - low) - (high - close)) /(high - low))
    ti['eom_{}'.format(14*mt)] = ta.volume.EaseOfMovementIndicator(high,low,volume,14*mt).sma_ease_of_movement()
    ti['mfi_{}'.format(14*mt)] = ta.volume.money_flow_index(high,low,close,volume,14*mt)
    ti['fi_{}'.format(1*mt)] = ta.volume.force_index(close,volume,1*mt)
    return ti

def get_ta_windows(df,mts):
    TA = []
    for mt in mts:
        TA.append(get_ta1(df,mt))
    TA = pd.concat(TA,axis=1)
    TA = TA.loc[:,~TA.columns.duplicated()]
    TA = TA.reindex(sorted(TA.columns),axis=1)
    return TA


