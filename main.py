
import streamlit as st
import datetime
from datetime import date

import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

import pandas as pd
import numpy as np

import math


def main():
    START = "2016-01-01"
    TODAY = date.today().strftime('%Y-%m-%d')

    st.title("Forecasting App")
    html_temp = """
        <div style="background-color:tomato;padding:10px">
        <h2 style="color:white;text-align:center;">This is a Stock Price Forecasting Application </h2>
        </div>
        """
    st.markdown(html_temp, unsafe_allow_html=True)

    stocks = ("AMZN","AAPL", "GOOG", "MSFT", "FB","CSCO","TSLA","SBUX")
    selected_stock = st.selectbox("Choose a stock for prediction: ", stocks)
    number_years = st.selectbox("Select the number of years for prediction: ", (1,2,3,4,5))
    period = number_years * 365

    data_load_State = st.text("Load data.....")

    @st.cache(allow_output_mutation=True)
    def loading_data(ticker):
        df1 = yf.download(ticker, START, TODAY)
        df1.reset_index(inplace=True)
        return df1
    df1 = loading_data(selected_stock)
    df1["Date"]=pd.to_datetime(df1['Date']).dt.date
    data_load_State.text("Loading data.......Done")

    st.subheader("Raw data")
    st.write(df1.tail())
    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df1['Date'], y=df1["Open"], name="Stock_Open"))
        fig.add_trace(go.Scatter(x=df1['Date'], y=df1["Close"], name="Stock_Close"))
        fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
    plot_raw_data()



    df_train = df1[["Date","Close"]]
    df_train =df_train.rename(columns={"Date":"ds","Close":"y"})
    model = Prophet()
    model.fit(df_train)
    pred = model.make_future_dataframe(periods=period)
    forecast = model.predict(pred)

    st.write("Forecsted Data")
    def plot_final_data():
        fig1 = plot_plotly(model,forecast)
        st.plotly_chart(fig1)
    plot_final_data()

    st.write("Forecast Componet")
    fig2 = model.plot_components(forecast)
    st.write(fig2)


# driver code
if __name__ == '__main__':
    main()