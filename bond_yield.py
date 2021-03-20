import requests
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
url = 'https://www.treasury.gov/resource-center/data-chart-center/interest-rates/Pages/TextView.aspx?data=yieldAll'
res = requests.get(url)
soup = BeautifulSoup(res.text,'html.parser')
b = ["".join(data.getText().split())for data in soup.find_all('td',{'class':'text_view_data'})]
cleaned = [b[i:i+13] for i in range(0, len(b), 13)]
df = pd.DataFrame(cleaned, columns = [h.getText() for h in soup.find_all('th')])
df.index = df['Date']
df = df.drop('Date', axis=1).replace('N/A',0).drop('10/11/10').drop('04/14/17').astype(float)
df.index = pd.to_datetime(df.index)
for i in range(0,len(df)):
    for d in range(0,12):
        try:
            if df.iloc[i][d] == 0 and df.iloc[i][d+1] != 0 :
                df.iloc[i][d] = df.iloc[i][d+1]
            elif df.iloc[i][d] == 0 and df.iloc[i][d+1] == 0 :
                df.iloc[i][d] = df.iloc[i][d+2]
        except:
            if df.iloc[i][d] == 0 and df.iloc[i][d-1] != 0 :
                df.iloc[i][d] = df.iloc[i][d-1]
            pass
# df=df.replace('N/A','nan')
Spread = df['10 yr']  - df['2 yr']
fig = plt.figure(figsize = (12,4))
ax = fig.add_subplot()

fig.suptitle('US Treasury Yield Spread, 10-Year Minus 2-Year', fontsize=16, fontweight='bold')
fig.subplots_adjust(top=0.85)
ax.set_title('1990/1/2 - 2021/03/05', fontsize=12, fontweight='bold')
ax.plot(Spread,linewidth=1,
         alpha=1,color='#1e609e')
ax.axhline(y=1.5, color="k", ls="--", alpha = 0.5)
ax.fill_between(Spread.index, 0, Spread,
                 where=Spread < 0, color="red",
                 alpha=0.8, interpolate=True),

ax.fill_between(Spread.index, 0, Spread,
                 where=Spread >= 0, color="#1e609e",
                 alpha=0.2, interpolate=True)
plt.yticks(np.arange(0, 4.2, 0.6))
ax.tick_params(axis="x", labelsize=10)
ax.tick_params(axis="y", labelsize=10)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xlim(left=df.index[0])
plt.xlim(right=df.index[-1])
plt.show()
plt.savefig('spread_result.png',dpi=300)
