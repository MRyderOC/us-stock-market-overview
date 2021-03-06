# us-stock-market-overview

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/mryderoc/us-stock-market-overview/main/MarketOverview.py)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/MRyderOC/us-stock-market-overview/main?urlpath=voila%2Frender%2FMarketOverview.ipynb)

US Stock Market data and insights.

Understand how different sectors and industries of the economy behave in the stock market.


This is a EDA (Exploratory Data Analysis) data science project that describes the stock market.

## Technologies

**Python 3.8**
#### You can find required libraries in requirements.txt.

## How to use
1. Clone the repo: ``` git clone "https://github.com/MRyderOC/us-stock-market-overview.git" ```.
2. Create a virtual environment: ```python3 -m venv myenv```.
3. Activate the virtual environment: ```source myenv/bin/activate```
4. Install dependencies: ```pip3 install -r requirements.txt```.
5. Run the app: 
   - ```streamlit run MarketOverview.py``` OR
   - ```voila MarketOverview.ipynb```
6. Enjoy!

> Use FinvizScrape.py to scrape today's data. It will store in the db folder: ```python3 FinvizScrape.py```

## Acknowledgments

* These scripts and notebooks were built using [finviz](https://finviz.com/screener.ashx) data.
* Thanks to [Mike](https://github.com/mtodisco10) who helped me with this project.

## Online view
You can find this app on streamlit cloud [here](https://share.streamlit.io/mryderoc/us-stock-market-overview/main/MarketOverview.py).
