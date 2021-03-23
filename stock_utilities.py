import pandas_datareader as pdr
import pandas as pd
from datetime import datetime,timedelta
from yahoo_fin import stock_info as si
import gc
from multiprocessing import Process, Manager,Queue

time_interval_pre = 7
time_interval_post = 5
days_hold = 3
profit_regret_upgap = 1
profit_regret_downgap = 5
# profit_wait_gap=2
profit_decay = 0.8

'''
Get the profit of holding the specified stock for days
'''
def get_potential(df,stock,sell_date,days_hold,profit_decay=0.8):
    while sell_date not in df.index:
        sell_date = sell_date + timedelta(days=1)
    value_sell=(df.loc[sell_date,'High']-df.loc[sell_date,'Low'])/2+df.loc[sell_date,'Low']
    # print('sell {} at {:.2f} on {}'.format(stock, value_sell,sell_date.strftime('%Y-%m-%d')))
    profit_potential=-1.5
    value_pre=value_sell
    date_current=sell_date
    ans=[]
    for i in range(days_hold):
        date_current=date_current+timedelta(days=1)
        while date_current not in df.index:
            date_current = date_current + timedelta(days=1)
        value_min_current = df.loc[date_current, 'Low']
        value_max_current = df.loc[date_current, 'High']
        profit_potential=profit_potential+100.0*(profit_decay**(i+1))*(value_pre-value_min_current)/value_pre-1
        value_pre=df.loc[date_current, 'Low']
        # print('Potential profit of {} after {} days is {:.2f}, current price is {:.2f}'.format(stock,i+1,profit_potential,value_pre))
        ans.append(profit_potential)
    return ans
"""
Evaluate the potential profit of the strategy
"""
def evaluate(stock,df,sell_date,time_interval_pre,profit_wait_gap):
    end_date = sell_date - timedelta(days=1)
    start_date = end_date - timedelta(days=time_interval_pre)
    value_min = df.loc[start_date:end_date, 'Low'].min()
    value_max = df.loc[start_date:end_date, 'High'].max()
    sell_decision = 0
    if (df.loc[start_date:end_date, 'Close'].values[-1] - value_min) / value_min > profit_wait_gap / 100:
        sell_decision = 1
    profits = get_potential(df, stock, sell_date, days_hold)
    profit_max = max(profits)
    profit_min = min(profits)
    sell_strong = 1 if min(profits) > profit_regret_downgap else 0
    dontsell_strong = 1 if max(profits) < profit_regret_upgap else 0
    shouldsell = sell_strong if sell_decision == 0 else 0
    shouldnotsell = dontsell_strong if sell_decision == 1 else 0
    # eval_matrix.append((stock, profit_wait_gap, sell_date.strftime('%Y-%m-%d'), profit_max, profit_min, sell_strong,
    #                     dontsell_strong, shouldsell, shouldnotsell, sell_decision))
    del df
    gc.collect()
    return (stock, profit_wait_gap, sell_date.strftime('%Y-%m-%d'), profit_max, profit_min, sell_strong,
                        dontsell_strong, shouldsell, shouldnotsell, sell_decision)
"""
Download the stock data.
Evaluate the performance of the strategy within period_ref
"""
def get_statistic(eval_matrixes, values_min,stock,period_init,period_ref):
    df = pdr.get_data_yahoo(stock,period_init[0], period_init[1])
    eval_matrix=[]
    profit_wait_gap=2
    for i in range(30):
        profit_wait_gap +=1
        sell_date = period_ref[0]
        while sell_date < period_ref[1] - timedelta(days=time_interval_post):
            eval_matrix.append(evaluate(stock,df,sell_date,time_interval_pre,profit_wait_gap))
            sell_date+=timedelta(days=1)
    sell_date=pd.to_datetime("today")
    end_date = sell_date - timedelta(days=1)
    start_date = end_date - timedelta(days=time_interval_pre)
    value_min = df.loc[start_date:end_date, 'Low'].min()
    eval_matrixes+=eval_matrix
    values_min.append((stock,value_min))
    print('Data collection of {} completed'.format(stock))


"""
Realtime prediction
"""
def predict_all_stocks(values_min,eval_df):

    eval_df.loc[eval_df['shouldnotsell'] == 0, 'prob_profit'] = eval_df['sell_decision'] * (
                eval_df['profit_max'] + eval_df['profit_min']) / 2
    eval_df.loc[eval_df['shouldnotsell'] != 0, 'prob_profit'] = eval_df['profit_min']
    eval_df_grouped=eval_df.groupby(['stock', 'wait_gap'])
    num_sell =eval_df_grouped['sell_decision'].sum()
    profit_total=eval_df_grouped['prob_profit'].sum()/num_sell
    rate_miss = eval_df_grouped['shouldsell'].mean()/num_sell
    rate_wrong = eval_df_grouped['shouldnotsell'].sum() / num_sell
    df_final = pd.merge(profit_total.rename('random'), rate_miss.rename('random'), on=['stock', 'wait_gap']).merge(
        rate_wrong.rename('random'), on=['stock', 'wait_gap']).merge(num_sell.rename('num_sell'),
                                                                         on=['stock', 'wait_gap'])
    df_final.columns = ['Expected profit', 'rate_miss', 'rate_wrong', 'num_sell']
    predict_matrix=[]
    for item in values_min.itertuples():
        stock=item[1]
        value_min=item[2]
        price_current = si.get_live_price(stock)
        increase=int((price_current-value_min)/value_min*100)
        print('Short stock {} at current price {:.2f}, increase {}'.format(stock,  price_current,
                                                                           increase))
        if increase > 32:
            increase = 32
        if increase < 3:
            increase = 3
        values = df_final.loc[(stock, increase), :].values
        # print(
        #     'Expected profit {:.2f}, miss rate {:.2f}, wrong rate {:.2f}, number of sell {}'.format(values[0], values[1],
        #                                                                                            values[2],
        #                                                                                            values[3]))
        predict_matrix.append([stock,price_current,increase, values[0], values[1], values[2],values[3]])
    predict_matrix=pd.DataFrame(predict_matrix,columns=['stock','price_cur','increase','profit','rate_miss','rate_wrong','sell_num'])
    predict_matrix=predict_matrix.sort_values(by=['profit','rate_wrong','price_cur','sell_num'],ascending=[False,True,True,False ])
    print(predict_matrix)
    return df_final,predict_matrix
