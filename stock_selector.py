import pandas_datareader as pdr
import pandas as pd
from datetime import datetime,timedelta
def get_potential(df,stock,buy_date,days_hold):
    while buy_date not in df.index:
        buy_date = buy_date + timedelta(days=1)
    value_buy=(df.loc[buy_date,'High']-df.loc[buy_date,'Low'])/2+df.loc[buy_date,'Low']
    # print('Buy {} at {:.2f} on {}'.format(stock, value_buy,buy_date.strftime('%Y-%m-%d')))
    profit_potential=-1.5
    value_pre=value_buy
    date_current=buy_date
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

time_interval_pre=7
days_hold=3
profit_regret_upgap=1
profit_regret_downgap=5
profit_wait_gap=10


stock_list = ['GME']#stock names
profit_decay=0.8
eval_indexes=['stock','wait_gap','date','profit_max','profit_min','buy_strong','dontbuy_strong','shouldbuy','shouldnotbuy','buy_decision']
eval_matrix=[]

for stock in stock_list:
    df = pdr.get_data_yahoo(stock,pd.to_datetime("2020-11-01"), pd.to_datetime("today"))
    profit_wait_gap=2
    for i in range(30):
        profit_wait_gap += 1
        buy_date = pd.to_datetime("2021-2-11")
        while buy_date < pd.to_datetime("today") - timedelta(days=7):
            end_date = buy_date - timedelta(days=1)
            start_date = end_date - timedelta(days=time_interval_pre)
            value_min=df.loc[start_date:end_date,'Low'].min()
            value_max=df.loc[start_date:end_date,'High'].max()
            buy_decision = 0
            if (df.loc[start_date:end_date,'Close'].values[-1]-value_min)/value_min>profit_wait_gap/100:
                buy_decision=1
            profits=get_potential(df, stock, buy_date, days_hold)
            profit_max=max(profits)
            profit_min=min(profits)
            buy_strong=1 if min(profits)>profit_regret_downgap else 0
            dontbuy_strong=1 if max(profits)<profit_regret_upgap else 0
            shouldbuy=buy_strong if buy_decision==0 else 0
            shouldnotbuy=dontbuy_strong if buy_decision==1 else 0
            eval_matrix.append((stock,profit_wait_gap,buy_date.strftime('%Y-%m-%d'),profit_max,profit_min,buy_strong,dontbuy_strong,shouldbuy,shouldnotbuy,buy_decision))
            buy_date+=timedelta(days=1)
eval_df=pd.DataFrame(eval_matrix, columns=eval_indexes)
eval_df.loc[eval_df['shouldnotbuy']==0,'prob_profit']=eval_df['buy_decision']*(eval_df['profit_max']+eval_df['profit_max'])/2
eval_df.loc[eval_df['shouldnotbuy']!=0,'prob_profit']=-5
profit_total=eval_df.groupby(['stock','wait_gap'])['prob_profit'].sum()/eval_df.groupby(['stock','wait_gap'])['buy_decision'].sum()
rate_miss=eval_df.groupby(['stock','wait_gap'])['shouldbuy'].mean()
rate_wrong=eval_df.groupby(['stock','wait_gap'])['shouldnotbuy'].sum()/eval_df.groupby(['stock','wait_gap'])['buy_decision'].sum()
df_final=pd.merge(profit_total.rename('Expected profit'),rate_miss,on=['stock','wait_gap']).merge(rate_wrong.rename('rate_wrong'), on=['stock','wait_gap'])
df_final.columns=['Expected profit','rate_miss','rate_wrong']
