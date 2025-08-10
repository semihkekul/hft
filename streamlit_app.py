from multiprocessing import util
import streamlit as st
from util import Util
from util import Model

import numpy as np
import pandas as pd
import altair as alt

st.set_page_config(
   page_title="HFT Simulation by Semih Kekül",
   page_icon="💱",
   layout="wide",
   initial_sidebar_state="expanded",
)

 
def on_generate_change():
    st.session_state.data.generate_value_change = True

def on_generate_click():
    st.session_state.data.generate_value_change = False
        # Simulate stock prices
    st.session_state.data.time, st.session_state.data.simulated_stock_prices = Util.simulate_stock_prices(st.session_state.data.S0, st.session_state.data.mu, st.session_state.data.sigma, st.session_state.data.T, st.session_state.data.dt)

    # Simulate market option prices with noise
    st.session_state.data.theoretical_prices  = np.array([Util.black_scholes_call(S, st.session_state.data.K, st.session_state.data.T - (day * st.session_state.data.dt), st.session_state.data.r, st.session_state.data.sigma) for day, S in zip(st.session_state.data.time, st.session_state.data.simulated_stock_prices)])
    st.session_state.data.market_prices = st.session_state.data.theoretical_prices + np.random.normal(0, st.session_state.data.noise, size=len(st.session_state.data.theoretical_prices))
    st.session_state.simple_model.profits = None

def on_model_value_change():
    st.session_state.data.model_value_change = True

def on_back_testing_click():
    st.session_state.data.model_value_change = False
    if model == "Simple":
        st.session_state.simple_model.profits, st.session_state.simple_model.signals = \
            Model.simple(st.session_state.data.market_prices, st.session_state.data.theoretical_prices, st.session_state.simple_model.threshold)
    elif model == "Model2":
        Model.model2()
    elif model == "Model3":
        Model.model3()

class SimpleModel:
    def __init__(self):
        self.threshold = 0.5
        self.profits = None
        self.signals = []

class Data:
    def __init__(self):
        self.simulated_stock_prices = None
        self.theoretical_prices = None
        self.market_prices = None
        self.time = None
        self.S0 = 100
        self.K = 100
        self.T = 1.0
        self.r = 0.01
        self.sigma = 0.2
        self.mu = 0.05
        self.generate_value_change = False
        self.noise = 0.5
        self.model_value_change = False

        

if 'data' not in st.session_state:
    st.session_state.data = Data()

if 'simple_model' not in st.session_state:
    st.session_state.simple_model = SimpleModel()


# Sidebar
st.sidebar.title("💱 HFT Simulation")

st.sidebar.markdown("Implemented by [Semih Kekül](https://www.linkedin.com/in/semihkekul/)")

st.sidebar.write("---")

st.sidebar.subheader("Geometric Brownian Motion and Black Scholes Model:")
st.session_state.data.S0 = st.sidebar.number_input("Initial stock price", min_value=0, value=st.session_state.data.S0, on_change=on_generate_change)
st.session_state.data.K = st.sidebar.number_input("Strike price", min_value=0, value=st.session_state.data.K, on_change=on_generate_change)
st.session_state.data.T = st.sidebar.number_input("Time to maturity (in years)", min_value=0.0, value=st.session_state.data.T, on_change=on_generate_change)
st.session_state.data.r = st.sidebar.number_input("Risk-free interest rate", min_value=0.0, value=st.session_state.data.r, on_change=on_generate_change)
st.session_state.data.sigma = st.sidebar.number_input("Volatility", min_value=0.0, value=st.session_state.data.sigma, on_change=on_generate_change)
st.session_state.data.mu = st.sidebar.number_input("Expected return", min_value=0.0, value=st.session_state.data.mu, on_change=on_generate_change)
st.session_state.data.dt = 1/252

st.session_state.data.noise = st.sidebar.number_input("Add noise to theoretical black scholes prices to get pseudo-real market prices", min_value=0.5, value=st.session_state.data.noise, on_change=on_generate_change)

generate_text = "💹 Generate"

if st.session_state.data.generate_value_change:
    generate_text += " *"
st.sidebar.button(generate_text, on_click=on_generate_click)

    

st.sidebar.write("---")
st.sidebar.subheader("HFT Model:")
model = st.sidebar.selectbox("Select Model:", ["Simple", "Model2", "Model3"])


if model == "Simple":
    st.sidebar.write("If the market price deviates from the theoretical price beyond a threshold, execute a trade.")
    st.session_state.simple_model.threshold = st.sidebar.number_input("Enter threshold:", value=st.session_state.simple_model.threshold,on_change=on_model_value_change)
elif model == "Model2" or model == "Model3":
    st.sidebar.write("Not implemented yet")


back_testing_text = "💹 Back Testing"
if st.session_state.data.generate_value_change or st.session_state.data.model_value_change:
    back_testing_text += " *"

st.sidebar.button(back_testing_text, on_click=on_back_testing_click, disabled = st.session_state.data.generate_value_change)



if st.session_state.data.simulated_stock_prices is not None:
    st.write("### Simulated Stock Prices")
    df = pd.DataFrame({
        "Theoretical Prices": st.session_state.data.theoretical_prices,
        "Market Prices": st.session_state.data.market_prices
    })
    st.line_chart(df, color=["#00FF00", "#0000FF"], x_label="Time (Days)", y_label="Price")

if st.session_state.simple_model.profits is not None:

    t = st.session_state.data.time
    profits = st.session_state.simple_model.profits
    signals = st.session_state.simple_model.signals
    df = pd.DataFrame({
        'Time': t,
        'Profit': profits,
        'Signal': signals
    })

    base = alt.Chart(df).mark_line(color='blue').encode(
        x=alt.X('Time', title='Time (Days)'),
        y=alt.Y('Profit', title='Cumulative Profit'),
        tooltip=['Time', 'Profit']
    )

    # Points for buy/sell signals
    signal_points = alt.Chart(df[df['Signal'].notnull()]).mark_point(filled=True).encode(
        x='Time',
        y='Profit',
        color=alt.Color('Signal', scale=alt.Scale(domain=['Buy', 'Sell'], range=['green', 'red'])),
        shape=alt.Shape('Signal', scale=alt.Scale(domain=['Buy', 'Sell'], range=['triangle-up', 'triangle-down'])),
        tooltip=['Time', 'Profit', 'Signal']
    )

    # Combine and save
    chart = (base + signal_points).properties(
        title='HFT Strategy: Profit Over Time with Buy/Sell Signals'
    )

    st.altair_chart(chart, theme=None)
