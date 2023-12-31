from bs4 import BeautifulSoup
from urllib.request import urlopen, Request
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
finviz_url="https://finviz.com/quote.ashx?t="

tickers=['AMZN','MSFT','GOOG']

news_tables={}
for ticker in tickers:
    url=finviz_url+ticker

    req=Request(url=url,headers={'user-agent' : 'my-app'})
    response=urlopen(req)
    
    html=BeautifulSoup(response, 'html')
    news_table=html.find(id="news-table")
    news_tables[ticker]=news_table
    

# tesla_data=news_tables['TSLA']
# tesla_rows=tesla_data.findAll('tr')


# for index,row in enumerate(tesla_rows):
#     title=row.a.text
#     timestamp=row.td.text
#     print(timestamp+" "+title)

bigger_data=[]

for ticker, news_table in news_tables.items():
    for row in news_table.findAll('tr'):
        cells = row.findAll('td')

        if len(cells) == 2:
            time_date_str = cells[0].text.strip()
            title = cells[1].a.text.strip()

            if ' ' in time_date_str:
                date, time = time_date_str.split(' ', 1)
            else:
                date = None
                time = time_date_str

            bigger_data.append([ticker, date, time, title])

df=pd.DataFrame(bigger_data,columns=['ticker','time','date','title'])

vader=SentimentIntensityAnalyzer()

df['compound'] = df['title'].apply(lambda title: vader.polarity_scores(title)['compound'])

# Convert date to datetime
df['date'] = pd.to_datetime(df['date']).dt.date

plt.figure(figsize=(4,6))
mean_df = df.groupby(['ticker', 'date']).mean()
# print(mean_df)
mean_df=mean_df.unstack()
mean_df = mean_df.xs('compound', axis="columns").transpose()
mean_df.plot(kind='bar')
plt.show()
