import pandas as pd 
import numpy as np 
from ta import momentum, trend, volatility, volume 
 

def get_ta(df_):
    """
    parameters
    df_ (pd.DataFrame): must contain [open,high,low,close,volume] in columns.
    """
    df = df_.copy()
    # assert error
    df.columns = [x.lower() for x in df.columns]
    

    df['m_rsi'] = momentum.rsi(df.close)
    df['m_roc'] = momentum.roc(df.close)
    df['m_cmo'] = get_cmo(df)
    df['m_wr']  = momentum.williams_r(df.high, df.low, df.close)
    df['vm_cmf'] = volume.chaikin_money_flow(df.high, df.low, df.close, df.volume)
    df['vm_mfi'] = volume.money_flow_index(df.high, df.low, df.close, df.volume)
    df['vm_fi'] = volume.force_index(df.close, df.volume)
    df['vm_eom'] = volume.ease_of_movement(df.high, df.low, df.volume)
    df['vl_bbp'] = volatility.bollinger_pband(df.close)
    df['vl_atr'] = volatility.average_true_range(df.high, df.low, df.close)
    #df['t_sma']  = trend.sma_indicator(df.close)
    df['t_macdd']  = trend.MACD(df.close).macd_diff()
    df['t_trix'] = trend.trix(df.close)
    df['t_cci'] = trend.cci(df.high, df.low, df.close)
    df['t_dpo'] = trend.dpo(df.close)  # trend occilator
    df['t_kst'] = trend.kst(df.close)  # know sure thing
    df['t_adx'] = trend.adx(df.high, df.low, df.close)  # adx:strength of trend
            
    return df 


def calculate_CMO(series): 
    sum_gains = series[series >= 0].sum()
    sum_losses = np.abs(series[series < 0].sum())
    cmo = 100 * ((sum_gains - sum_losses) / (sum_gains + sum_losses))
    
    return np.round(cmo, 3)

def get_cmo(df, period=13):
    #tmp = df.close.diff().rolling(period).apply(calculate_CMO, args=(period,), raw=True)
    
    tmp = df.close.diff().rolling(period).apply(calculate_CMO, raw=True)
    return tmp

