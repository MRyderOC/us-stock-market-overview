from itertools import combinations
import json
import time
import datetime

import numpy as np
import pandas as pd
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


def read_data(path: str = '') -> pd.DataFrame:
    if not path:
        raise ValueError('No path provided.')
    df = pd.read_csv(path)
    df = df.set_index('No.')
    return df

@st.cache
def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy(deep=True)
    # Replacing missing data with NaN
    for col in df.columns:
        df[col].replace('-', np.nan, inplace=True)

    # Price column: Data Cleaning
    def price_calc(row):
        return np.nan if row['Price'] is np.nan else float(row['Price'])
    df['Price'] = df.apply(price_calc, axis='columns')

    # Change column: Data Cleaning
    def change_calc(row):
        return np.nan if row['Change'] is np.nan else float(str(row['Change'])[:-1])
    df['Change'] = df.apply(change_calc, axis='columns')

    # Volume column: Data Cleaning
    def vol_calc(row):
        return np.nan if row['Volume'] is np.nan else int(''.join(str(row.Volume).split(',')))
    df['Volume'] = df.apply(vol_calc, axis='columns')

    # Market Cap: Data Cleaning
    def cap_calc(row):
        s = str(row['Market Cap']).strip()
        if s[-1] == 'M':
            return float(s[:-1])
        elif s[-1] == 'B':
            return float(s[:-1]) * 1000
    # Notify that Market Cap is based on Million $.
    df['Market Cap'] = df.apply(cap_calc, axis='columns')

    # P/E column: Data Cleaning
    def pe_calc(row):
        return float(row['P/E'])
    df['P/E'] = df.apply(pe_calc, axis='columns')

    # Renaming columns for ease of use
    df.rename(columns={col: '_'.join(col.split()).lower() for col in df.columns}, inplace=True)

    return df

@st.cache
def stocks_data(data: pd.DataFrame) -> pd.DataFrame:
    df = data[data['industry'] != 'Exchange Traded Fund'].reset_index()
    df.drop(['p/e', 'No.'], axis=1, inplace=True)
    df.dropna(subset=['market_cap'], inplace=True)
    return df

@st.cache
def etf_data(data: pd.DataFrame) -> pd.DataFrame:
    df = data[data['industry'] == 'Exchange Traded Fund'].reset_index()
    df.drop(['No.', 'p/e', 'market_cap', 'country', 'industry', 'sector'], axis=1, inplace=True)
    df.dropna(subset=['price'], inplace=True)
    return df

def region_data(stocks_df: pd.DataFrame) -> pd.DataFrame:

    def findRegion(row):
        Regions = {
            'America': [
                'Uruguay', 'Peru', 'Panama',
                'Mexico', 'Costa Rica', 'Colombia',
                'Chile', 'Brazil', 'Bahamas', 'Argentina'
            ],
            'Europe': [
                'Belgium', 'Bermuda', 'Cayman Islands', 'Cyprus',
                'Denmark', 'Finland', 'France', 'Germany', 'Greece',
                'Ireland', 'Italy', 'Luxembourg', 'Malta', 'Monaco',
                'Netherlands', 'Norway', 'Russia', 'Spain', 'Sweden',
                'Switzerland', 'Turkey', 'United Kingdom'
            ],
            'Asia': [
                'Taiwan', 'South Korea', 'Singapore',
                'Philippines', 'Kazakhstan', 'Japan',
                'Indonesia', 'India', 'Hong Kong',
                'Israel', 'United Arab Emirates'
            ],
            'Other': [
                'Australia', 'New Zealand', 'South Africa'
            ],
            'USA': ['USA'],
            'Canada': ['Canada'],
            'China': ['China'],
        }
        for region in Regions:
            if row['country'] in Regions[region]:
                return region

    df = stocks_df.copy(deep=True)
    df['region'] = df.apply(findRegion, axis='columns')
    return df

@st.cache
def read_historical(ticker_list: list) -> dict:
    historical_data = {}
    for ticker in ticker_list:
        try:
            historical_data[ticker] = pd.read_csv(f'./db/stocks_data/{ticker}.csv')
        except Exception as e:
            print(f'Ticker {ticker} missed because: {e}')
    return historical_data


def find_intersection_date(ticker1, ticker2):
    return max(ticker1.index[0], ticker2.index[0])

def correlation_finder(
    ticker1: pd.DataFrame,
    ticker2: pd.DataFrame,
    column: str = 'Close'
) -> float:
    '''
    Finding correlation of two tickers on a specific column.

    column acceptable values: Close, Open, High, Low
    '''
    if column not in ['Close', 'Open', 'Low', 'High']:
        raise ValueError('Column value invalid')
    try:
        time = find_intersection_date(ticker1, ticker2)
        return ticker1.loc[time:][column].corr(ticker2.loc[time:][column])
    except Exception:
        return ticker1[column].corr(ticker2[column])

def correlation_batch_finder(historical_data: dict) -> dict:
    """
    Pairwise correlation on group of tickers.

    Parameters
    ----------
    historical_data: dict
        A dictionary that keys are ticker names as str and values are DataFrame
    """
    # for ticker in historical_data.values():
    #     if not isinstance(ticker, pd.core.frame.DataFrame):
    #         raise TypeError('Inputs are not pandas DataFrame')

    out = {}
    for ticker1, ticker2 in combinations(historical_data, 2):
        c = correlation_finder(
            historical_data[ticker1],
            historical_data[ticker2]
        )
        if np.isnan(c):
            continue
        else:
            out[ticker1 + ', ' + ticker2] = c
    return out

def correlation_batch_threshold(batch_correlation: dict, threshold: float) -> json:
    '''Finding pairs which is exceeds threshold.'''
    return json.dumps({
        key: value
        for key, value in sorted(
            batch_correlation.items(),
            key=lambda item: abs(item[1]),
            reverse=True
        )
        if value >= threshold
    })

def correlation_batch_top_k(batch_correlation: dict, k: int = 3) -> json:
    """
    Find top k correlations in the batch_correlation dictionary
    that has the structure: (ticker1, ticker2): correlation.

    Parameters
    ----------
    batch_correlation: dict
        A dictionary with structure keys as tuple and values as float.
    k: int; default = 3
        Specify how many top elements should return.
    """
    if not batch_correlation:
        raise ValueError('batch_correlation can not be empty.')
    if k <= 0:
        raise ValueError('k can not accept negative values.')
    return json.dumps({
        key: value
        for key, value in sorted(
            batch_correlation.items(),
            key=lambda item: abs(item[1]),
            reverse=True
        )[:k]
    })


def general_app(df: pd.DataFrame):
    st.header('General')
    st.subheader('Main DataFrame')
    st.dataframe(df)
    st.markdown(f"""
        - Market Cap missing {
            df.isnull().sum().market_cap
        } rows
            - {
                df[df.market_cap.isnull()].sector.value_counts().Financial
            } of missing data belong to Financial Sector
                - {
                    df[df.market_cap.isnull()].industry.value_counts()['Exchange Traded Fund']
                } of missing data belong to ETFs in Financial Sector
                - {
                    df[df.market_cap.isnull()].industry.value_counts()['Shell Companies']
                } of missing data belong to Shell Companies in Financial Sector (Almost {
                    round(
                        100
                        * df[df.market_cap.isnull()].industry.value_counts()['Shell Companies']
                        / len(df[df['industry'] == 'Shell Companies']),
                        -1
                    )
                }% of Shell Companies)
                - {
                    df[df.market_cap.isnull()].sector.value_counts().Financial
                    - df[df.market_cap.isnull()].industry.value_counts()['Exchange Traded Fund']
                    - df[df.market_cap.isnull()].industry.value_counts()['Shell Companies']
                } of missing data belong to other Industries in Financial Sector
            - {
                df.isnull().sum().market_cap
                - df[df.market_cap.isnull()].sector.value_counts().Financial
            } of missing data belong to other Sectors

        > The most missing data from Market Cap column belongs to ETFs.

        Since ETFs are real important part,
        we can separate our analysis into 2 different parts: Non ETFs(Stocks) & ETFs
    """)

def stocks_app(stocks_df: pd.DataFrame):
    st.header('Stocks')
    st.dataframe(stocks_df)

    st.subheader("Intuition")
    st.caption(
        "We need to look at the data and it's descriptive measurements"
        " to have some insight with the data we're dealing with."
    )
    st.table(stocks_df.describe())

    st.subheader("Let's take a look at correlation matrix:")
    cor = stocks_df.corr()
    fig, ax = plt.subplots()
    ax = sns.heatmap(cor, xticklabels=cor.columns, yticklabels=cor.columns, annot=True)
    st.pyplot(fig)
    correlation_set = set(
        tuple(sorted([col, cor.columns[ind]]))
        for col in cor.columns
        for ind in np.where((cor[col] >= 0.1) & (cor[col] != 1))[0]
    )
    st.caption(
        "> It seems there is a little correlation between:\n"
        + '\n'.join([
            '- **' + col1 + '**' + ', ' + '**' + col2 + '**'
            for col1, col2 in correlation_set
        ])
    )

def etf_app(etf_df: pd.DataFrame):
    st.header('ETF (Exchange Traded Fund)')
    st.dataframe(etf_df)

    st.subheader("Intuition")
    st.caption(
        "We need to look at the data and it's descriptive measurements"
        " to have some insight with the data we're dealing with."
    )
    st.table(etf_df.describe())

    st.subheader("Let's take a look at correlation matrix:")
    cor_etf = etf_df.corr()
    fig, ax = plt.subplots()
    ax = sns.heatmap(cor_etf, xticklabels=cor_etf.columns, yticklabels=cor_etf.columns, annot=True)
    st.pyplot(fig)
    correlation_set_etf = set(
        tuple(sorted([col, cor_etf.columns[ind]]))
        for col in cor_etf.columns
        for ind in np.where((cor_etf[col] >= 0.1) & (cor_etf[col] != 1))[0]
    )
    st.caption(
        "> It seems there is a little correlation between:\n"
        + '\n'.join([
            '- **' + col1 + '**' + ', ' + '**' + col2 + '**'
            for col1, col2 in correlation_set_etf
        ])
    )

def top_movers_stocks_app(stocks_df: pd.DataFrame):
    st.header("Top movers")

    st.subheader("Most traded stocks")
    df = stocks_df.reset_index()
    zs_vol = zscore(stocks_df.volume)
    volume_outliers = np.where(abs(zs_vol) > 3)[0]
    st.dataframe(
        df.loc[volume_outliers].sort_values(
            by=['volume'],
            ascending=False
        ).reset_index().drop(['index', 'level_0'], axis=1)
    )

    st.subheader("Most percentage change")
    zs_change = zscore(stocks_df.change)
    change_outliers = np.where(abs(zs_change) > 4)[0]
    st.dataframe(
        df.loc[change_outliers].sort_values(
            by=['change'],
            key=lambda col: abs(col),
            ascending=False
        ).reset_index().drop(['index', 'level_0'], axis=1)
    )

    st.subheader("Most traded and most changed")
    both_outliers = sorted(list(set(volume_outliers).intersection(change_outliers)))
    st.dataframe(
        df.loc[both_outliers].sort_values(
            by=['change'],
            ascending=False
        ).reset_index().drop(['index', 'level_0'], axis=1)
    )

def top_movers_etf_app(etf_df: pd.DataFrame):
    st.header("Top movers")

    st.subheader("Most traded ETFs")
    # df = etf_df.reset_index()
    zs_vol = zscore(etf_df.volume)
    volume_outliers = np.where(abs(zs_vol) > 3)[0]
    st.dataframe(
        etf_df.loc[etf_df.index.intersection(volume_outliers)].sort_values(
            by=['volume'],
            ascending=False
        ).reset_index().drop(['index'], axis=1)
    )

    st.subheader("Most percentage change")
    zs_change = zscore(etf_df.change)
    change_outliers = np.where(abs(zs_change) > 4)[0]
    st.dataframe(
        etf_df.loc[etf_df.index.intersection(change_outliers)].sort_values(
            by=['change'],
            key=lambda col: abs(col),
            ascending=False
        ).reset_index().drop(['index'], axis=1)
    )

    st.subheader("Most traded and most changed")
    both_outliers = sorted(list(set(volume_outliers).intersection(change_outliers)))
    st.dataframe(
        etf_df.loc[etf_df.index.intersection(both_outliers)].sort_values(
            by=['change'],
            ascending=False
        ).reset_index().drop(['index'], axis=1)
    )

def group_analysis_sector_app(stocks_df: pd.DataFrame):
    st.header("Group Analysis: Sector")
    sector_group = stocks_df.groupby('sector')

    sectors = sorted(stocks_df['sector'].unique())
    colors = [
        'grey', 'red', 'green',
        'orange', 'olive', 'gold',
        'orchid', 'steelblue', 'mediumpurple',
        'teal', 'pink',
    ]
    with st.expander('Number of Companies'):
        st.subheader('Number of Companies')
        chart_number_of_companies = st.radio(
            'Choose your chart: Number of Companies',
            ['matplotlib chart', 'streamlit chart']
        )
        if chart_number_of_companies == 'streamlit chart':
            st.bar_chart(sector_group.ticker.count(), use_container_width=False)
            st.dataframe(sector_group.ticker.count())
        elif chart_number_of_companies == 'matplotlib chart':
            fig, ax = plt.subplots(figsize=(3, 3))
            plt.bar(
                np.arange(len(sector_group.ticker.count())),
                sector_group.ticker.count(),
                color=colors
            )
            plt.xticks(np.arange(len(sector_group.ticker.count())), sectors, rotation=90)
            plt.title("Sector comparison by:  Sector's total # of companies")
            st.pyplot(fig)

    with st.expander('Volume'):
        st.subheader('Volume')
        chart_volume = st.radio(
            'Choose your chart: Volume',
            ['matplotlib chart', 'streamlit chart']
        )
        if chart_volume == 'streamlit chart':
            st.bar_chart(sector_group.volume.sum(), use_container_width=False)
            st.dataframe(sector_group.volume.sum())
        elif chart_volume == 'matplotlib chart':
            fig, ax = plt.subplots(figsize=(3, 3))
            plt.bar(
                np.arange(len(sector_group.volume.sum())),
                sector_group.volume.sum(),
                color=colors
            )
            plt.xticks(np.arange(len(sector_group.volume.sum())), sectors, rotation=90)
            plt.title("Sector comparison by:  Sector's total volume traded")
            st.pyplot(fig)

    with st.expander('Market Cap'):
        st.subheader('Market Cap')
        chart_market_cap = st.radio(
            'Choose your chart: Market Cap',
            ['matplotlib chart', 'streamlit chart']
        )
        if chart_market_cap == 'streamlit chart':
            st.bar_chart(sector_group.market_cap.sum(), use_container_width=False)
            st.dataframe(sector_group.market_cap.sum())
        elif chart_market_cap == 'matplotlib chart':
            fig, ax = plt.subplots(figsize=(3, 3))
            plt.bar(
                np.arange(len(sector_group.market_cap.sum())),
                sector_group.market_cap.sum(),
                color=colors
            )
            plt.xticks(np.arange(len(sector_group.market_cap.sum())), sectors, rotation=90)
            plt.title("Sector comparison by:  Sector's total Market Cap")
            st.pyplot(fig)

    with st.expander('-- Conclusion --'):
        st.markdown(
            """
            Major things to consider from these plots are:
            - ***Basic Materials:***
            Number of companies and market cap are low BUT the volume traded is high.

            - ***Financial:***
            Number of companies is really high and the market cap is considerable
            BUT volume traded is low.

            - ***Healthcare:***
            Most traded sector based on volume with large number of companies
            BUT the market cap is not high enough.

            - ***Technology:***
            It has the largest market cap with average number of companies AND good volume trading.

            - ***Communication Services / Consumer Cyclical:***
            Even though the number of companies is low BUT the market cap is high.

            - ***Consumer Defensive / Energy / Industrials / Real Estate / Utilities:***
            These are most likely the same in all three plots.
            """
        )

def group_analysis_industry_app(stocks_df: pd.DataFrame):
    st.header("Group Analysis: Industry")
    sector_group = stocks_df.groupby('sector')
    industry_group = stocks_df.groupby(['sector', 'industry'])

    with st.expander('Sector / Industry charts'):
        pie_chart_type = st.radio(
            "Chart based on",
            ['Number of companies', 'Volume', 'Market Cap']
        )
        colors = [
            'grey', 'red', 'green',
            'orange', 'olive', 'gold',
            'orchid', 'steelblue', 'mediumpurple',
            'teal', 'pink',
        ]
        if pie_chart_type == 'Number of companies':
            fig, ax = plt.subplots()
            plt.pie(
                sector_group.ticker.count(),
                labels=sector_group.ticker.count().index,
                colors=colors,
                wedgeprops=dict(width=0.3, edgecolor='w'),
                radius=1.1,
                autopct='%1.1f%%',
                pctdistance=0.9,
                rotatelabels=True,
            )
            plt.pie(industry_group.ticker.count(), radius=0.8)
            plt.title("Industry comparison grouped by Sectors: total # of companies", y=1.3)
            st.pyplot(fig)
            st.markdown(
                """
                - Industries with considerable number of companies:
                    - Couple of industries in Financial sector
                    - One industry in Technology sector
                    - One industry in Healthcare sector
                """
            )
        elif pie_chart_type == 'Volume':
            fig, ax = plt.subplots()
            plt.pie(
                sector_group.volume.sum(),
                labels=sector_group.volume.sum().index,
                colors=colors,
                wedgeprops=dict(width=0.3, edgecolor='w'),
                radius=1.1,
                autopct='%1.1f%%',
                pctdistance=0.9,
                rotatelabels=True
            )
            plt.pie(industry_group.volume.sum(), radius=0.8)
            plt.title("Industry comparison grouped by Sectors: total volume traded", y=1.3)
            st.pyplot(fig)
            st.markdown(
                """
                - Industries with considerable volume traded:
                    - Two major industries in Healthcare sector
                    - Two major industries in Consumer Cyclical sector
                    - One industry in Technology sector
                    - One industry in Basic Materails sector
                    - Three major industries in Communication Services sector
                    - One industry in Industrials sector
                """
            )
        elif pie_chart_type == 'Market Cap':
            fig, ax = plt.subplots()
            plt.pie(
                sector_group.market_cap.sum(),
                labels=sector_group.market_cap.sum().index,
                colors=colors,
                wedgeprops=dict(width=0.3, edgecolor='w'),
                radius=1.1,
                autopct='%1.1f%%',
                pctdistance=0.9,
                rotatelabels=True
            )
            plt.pie(industry_group.market_cap.sum(), radius=0.8)
            plt.title("Industry comparison grouped by Sectors: total Market Cap", y=1.3)
            st.pyplot(fig)
            st.markdown(
                """
                - Industries with considerable market cap:
                    - Couple of industries in Technology sector
                    - One industry in Communication Services sector
                    - One industry in Consumer Cyclical sector
                    - One industry in Healthcare sector
                """
            )

    st.markdown("### Let's look at the industries in each sector for more intuition.")

    sectors = sorted(stocks_df['sector'].unique())
    sector_choice = st.selectbox('Choose the Sector', sectors)
    col1, col2 = st.columns(2)
    with col1:
        industry_criterion = st.radio(
                'Choose your criterion:',
                ['Number of Companies', 'Volume', 'Market Cap']
            )
    with col2:
        chart_industry = st.radio(
                'Choose your chart:',
                ['matplotlib chart', 'streamlit chart']
            )

    if industry_criterion == 'Number of Companies':
        plot_data = sector_group.get_group(sector_choice).groupby('industry').ticker.count()
    elif industry_criterion == 'Volume':
        plot_data = sector_group.get_group(sector_choice).groupby('industry').volume.sum()
    elif industry_criterion == 'Market Cap':
        plot_data = sector_group.get_group(sector_choice).groupby('industry').market_cap.sum()

    if chart_industry == 'streamlit chart':
        st.bar_chart(plot_data, use_container_width=False)
        st.dataframe(plot_data)
    elif chart_industry == 'matplotlib chart':
        fig, ax = plt.subplots(figsize=(3, 3))
        plt.title(f'Industry comparison in sector " {sector_choice} ": total {industry_criterion}')
        plt.bar(
            np.arange(len(plot_data)),
            plot_data,
            color=sns.color_palette('muted', n_colors=25)
        )
        plt.xticks(
            np.arange(len(plot_data)),
            plot_data.index,
            rotation=90
        )
        st.pyplot(fig)

def group_analysis_region_app(stocks_df: pd.DataFrame):
    region_df = region_data(stocks_df)
    region_group = region_df.groupby('region')
    st.subheader("Group Analysis: Region")
    st.caption(
        "> Since the major countries in the market are USA, Canada, and China,"
        " we separate these countries from other regions."
    )

    with st.expander('Number of Companies'):
        st.subheader('Number of Companies')
        st.bar_chart(region_group.ticker.count())
        st.dataframe(region_group.ticker.count())

    with st.expander('Volume'):
        st.subheader('Volume')
        st.bar_chart(region_group.volume.sum())
        st.dataframe(region_group.volume.sum())

    with st.expander('Market Cap'):
        st.subheader('Market Cap')
        st.bar_chart(region_group.market_cap.sum())
        st.dataframe(region_group.market_cap.sum())

    with st.expander('-- Conclusion --'):
        st.markdown(
            """
            Major things to consider from these plots are:
            - Still ***USA*** dominate the market.
            - Even tough ***Canadian***
            companies traded with high volume, their market cap is not much.
            - ***Europe*** and ***China***
            act similar to each other but Europe has a stronger impact
            both in volume and market cap.
            """
        )

def group_analysis_country_app(stocks_df: pd.DataFrame):
    st.header("Group Analysis: Country")
    country_group = stocks_df.groupby('country')

    with st.expander('Number of Companies'):
        st.subheader('Number of Companies')
        st.bar_chart(country_group.ticker.count())
        st.dataframe(country_group.ticker.count())

    with st.expander('Volume'):
        st.subheader('Volume')
        st.bar_chart(country_group.volume.sum())
        st.dataframe(country_group.volume.sum())

    with st.expander('Market Cap'):
        st.subheader('Market Cap')
        st.bar_chart(country_group.market_cap.sum())
        st.dataframe(country_group.market_cap.sum())

    with st.expander('-- Conclusion --'):
        st.markdown(
            """
            It's obvious that USA dominate the market. The leading countries are: USA, Canada, China

            To get a better intuition,
            It's recommended to look at the data by region.
            To do so, we're going to add a column Region.
            """
        )
    st.markdown("---")

    group_analysis_region_app(stocks_df)

def correlation_app(stocks_df: pd.DataFrame):
    s = time.time()
    st.header('Correlation: Sector')
    st.markdown(
        """
        Find most correlated tickers in specific industries.

        > Each part takes a little time since there are several tickers in each industry.
        **Thanks for your patience.**
        """
    )

    sector_choice = st.selectbox(
        'Choose your desired sector',
        sorted(stocks_df['sector'].unique())
    )
    sector_df = stocks_df[stocks_df['sector'] == sector_choice]

    col1, col2 = st.columns(2)
    with col1:
        industry_choice = st.radio(
            'Choose your desired industry:',
            sector_df['industry'].unique()
        )
    with col2:
        top_threshold = st.radio(
            'Choose representation',
            ['Pick top k', 'Pick threshold']
        )

    industry_df = sector_df[sector_df['industry'] == industry_choice]
    industry_historical_data = read_historical(list(industry_df['ticker']))
    industry_correlation_dict = correlation_batch_finder(industry_historical_data)
    if top_threshold == 'Pick top k':
        k = st.slider('Pick top k correlated tickers: ', 1, 20, 3)
        st.json(correlation_batch_top_k(industry_correlation_dict, k))
    elif top_threshold == 'Pick threshold':
        threshold = st.slider('Pick threshold for correlation: ', 0.0, 1.0, 0.97)
        st.json(correlation_batch_threshold(industry_correlation_dict, threshold))

    print(f'Time is: {time.time() - s}')


def whole_st_app():
    """Gather the whole app together."""
    yesterday = datetime.datetime.today() - datetime.timedelta(days=1)
    try:
        raw_df = read_data(f'./db/{str(yesterday).split()[0]}_raw.csv')
        yesterday_flag = True
    except FileNotFoundError:
        raw_df = read_data('./db/2021-09-24_raw.csv')
        yesterday_flag = False
    clean_df = clean_data(raw_df)
    stocks_df = stocks_data(clean_df)
    etf_df = etf_data(clean_df)

    st.title("US Stock Market Overview")
    st.markdown(
        f"""
        This app goes through the US stock market and
        shows some of the main criteria of the market.

        The data gathered from [finviz.com](https://finviz.com/screener.ashx) in {
            yesterday.strftime("%b/%d/%Y") if yesterday_flag else "Sep/24/2021"
        }.

        ---
        """
    )
    menu = [
        'General',
        'Stocks',
        'ETFs',
        'Top Movers: Stocks',
        'Top Movers: ETFs',
        'Group Analysis: Sector',
        'Group Analysis: Industry',
        'Group Analysis: Country',
        'Correlation',
    ]
    menu_choice = st.sidebar.selectbox('Menu', menu)

    if menu_choice == 'General':
        general_app(clean_df)
    elif menu_choice == 'Stocks':
        stocks_app(stocks_df)
    elif menu_choice == 'ETFs':
        etf_app(etf_df)
    elif menu_choice == 'Top Movers: Stocks':
        top_movers_stocks_app(stocks_df)
    elif menu_choice == 'Top Movers: ETFs':
        top_movers_etf_app(etf_df)
    elif menu_choice == 'Group Analysis: Sector':
        group_analysis_sector_app(stocks_df)
    elif menu_choice == 'Group Analysis: Industry':
        group_analysis_industry_app(stocks_df)
    elif menu_choice == 'Group Analysis: Country':
        group_analysis_country_app(stocks_df)
    elif menu_choice == 'Correlation':
        correlation_app(stocks_df)

if __name__ == '__main__':
    whole_st_app()