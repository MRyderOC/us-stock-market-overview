import numpy as np
import pandas as pd
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
    def priceCalc(row):
        return np.nan if row['Price'] is np.nan else float(row['Price'])
    df['Price'] = df.apply(priceCalc, axis='columns')

    # Change column: Data Cleaning
    def changeCalc(row):
        return np.nan if row['Change'] is np.nan else float(str(row['Change'])[:-1])
    df['Change'] = df.apply(changeCalc, axis='columns')

    # Volume column: Data Cleaning
    def volCalc(row):
        return np.nan if row['Volume'] is np.nan else int(''.join(str(row.Volume).split(',')))
    df['Volume'] = df.apply(volCalc, axis='columns')

    # Market Cap: Data Cleaning
    def capCalc(row):
        s = str(row['Market Cap']).strip()
        if s[-1] == 'M':
            return float(s[:-1])
        elif s[-1] == 'B':
            return float(s[:-1]) * 1000
    # Notify that Market Cap is based on Million $.
    df['Market Cap'] = df.apply(capCalc, axis='columns')

    # P/E column: Data Cleaning
    def peCalc(row):
        return float(row['P/E'])
    df['P/E'] = df.apply(peCalc, axis='columns')

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
        )
    )

    st.subheader("Most percentage change")
    zs_change = zscore(stocks_df.change)
    change_outliers = np.where(abs(zs_change) > 4)[0]
    st.dataframe(
        df.loc[change_outliers].sort_values(
            by=['change'],
            key=lambda col: abs(col),
            ascending=False
        )
    )

    st.subheader("Most traded and most changed")
    both_outliers = sorted(list(set(volume_outliers).intersection(change_outliers)))
    st.dataframe(df.loc[both_outliers].sort_values(by=['change'], ascending=False))

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
        )
    )

    st.subheader("Most percentage change")
    zs_change = zscore(etf_df.change)
    change_outliers = np.where(abs(zs_change) > 4)[0]
    st.dataframe(
        etf_df.loc[etf_df.index.intersection(change_outliers)].sort_values(
            by=['change'],
            key=lambda col: abs(col),
            ascending=False
        )
    )

    st.subheader("Most traded and most changed")
    both_outliers = sorted(list(set(volume_outliers).intersection(change_outliers)))
    st.dataframe(
        etf_df.loc[etf_df.index.intersection(both_outliers)].sort_values(
            by=['change'],
            ascending=False
        )
    )




def whole_st_app():
    """Gather the whole app together."""
    path = '2021-09-11T11:49:29.csv'
    raw_df = read_data(path)
    clean_df = clean_data(raw_df)
    stocks_df = stocks_data(clean_df)
    etf_df = etf_data(clean_df)

    st.title("US Stock Market Overview")
    st.markdown(
        """
        This app goes through the US stock market and
        shows some of the main criteria of the market.

        The data gathered from [finviz.com](https://finviz.com/screener.ashx).

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



if __name__ == '__main__':
    whole_st_app()