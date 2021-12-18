from urllib.request import urlopen, Request
import requests
from bs4 import BeautifulSoup
import regex as re
import numpy as np
import multiprocessing as mp
import os
import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import uuid


def get_articles_text(tick):
    finwiz_url = 'https://finviz.com/quote.ashx?t='
    text = []
    news_tables = {}
    url = finwiz_url + tick
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.93 Safari/537.36'}      #'referer': 'https://ogs.google.com/', 'authority': 'play.google.com'
    req = Request(url=url,
                  headers=headers)
    response = urlopen(req)
    # Read the contents of the file into 'html'
    html = BeautifulSoup(response, features="html.parser")
    # Find 'news-table' in the Soup and load it into 'news_table'
    news_table = html.find(id='news-table')#'news-table'
    news_tables[tick] = news_table

    filename = uuid.uuid4().hex
    f = open(filename, 'w')
    f.write(str(news_table))
    f.close()
    f = open(filename, 'r')
    lines = f.read()
    f.close()
    os.remove(filename)

    dates = []
    for file_name, news_table in news_tables.items():
        # Iterate through all tr tags in 'news_table'
        for x in news_table.findAll('tr'):
            date_scrape = x.td.text.split()

            if len(date_scrape) == 1:
                time = date_scrape[0]

            else:
                date = date_scrape[0]
                time = date_scrape[1]
            dates.append(pd.to_datetime(date).date())

    separated_article_links = re.findall(r'(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])', lines)
    article_links = []

    for i in range(len(separated_article_links)):
        link = separated_article_links[i][0] + '://' + separated_article_links[i][1] + separated_article_links[i][2]
        article_links.append([dates[i], link])
    #print(article_links)

    for date, link in article_links:
        article = requests.get(link, headers=headers)
        article_content = article.content
        soup_article = BeautifulSoup(article_content, features='html.parser')
        body = soup_article.find_all('p')

        if not bool(body):
            continue

        # Unifying the paragraphs
        list_paragraphs = []
        for p in np.arange(0, len(body)):
            paragraph = body[p].get_text()
            list_paragraphs.append(paragraph)
            final_article = " ".join(list_paragraphs)
        text.append([tick, date, final_article])

    return text


def sentiment_analysis(text):
    analyzer = SentimentIntensityAnalyzer()
    columns = ['ticker', 'date', 'article']
    scored_news = pd.DataFrame(text, columns=columns)
    today = pd.to_datetime('today').date()
    scored_news = scored_news.loc[(scored_news['date'] >= (today - pd.Timedelta(days=14)))]
    #print(scored_news)
    scores = scored_news['article'].apply(analyzer.polarity_scores).tolist()
    scores_df = pd.DataFrame(scores)
    scored_news = scored_news.join(scores_df, rsuffix='_right')
    scored_news['date'] = pd.to_datetime(scored_news.date).dt.date

    return scored_news


def chart_scores(scores):
    combined_scores = pd.concat(scores)
    daily_mean_scores = combined_scores.groupby(['ticker', 'date']).mean()
    daily_mean_scores = daily_mean_scores.unstack()
    daily_mean_scores = daily_mean_scores.xs('compound', axis='columns').transpose()

    plt.rcParams['figure.figsize'] = [20, 12]
    daily_mean_scores.plot(kind='bar')
    plt.grid()
    plt.show(aspect='auto')


def main():
    tickers = input("Enter tickers of company/companies (separated by a space) you want to see the sentiment of: ")
    tickers = tickers.split()
    print(tickers)
    pool = mp.Pool(mp.cpu_count())
    text_result = pool.map(get_articles_text, tickers)
    score_result = pool.map(sentiment_analysis, text_result)
    chart_scores(score_result)


if __name__ == "__main__":
    main()
