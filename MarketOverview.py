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




if __name__ == '__main__':
    whole_st_app()