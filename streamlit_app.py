import streamlit as st
from util import Util

st.title("💱HFT 💹")


# Sidebar 
st.sidebar.header("Options")
st.sidebar.subheader("Generate Historical Data")
S0 = st.sidebar.number_input("Initial stock price", min_value=0, value=100)
K = st.sidebar.number_input("Strike price", min_value=0, value=100)
T = st.sidebar.number_input("Time to maturity (in years)", min_value=0.0, value=1.0)
r = st.sidebar.number_input("Risk-free interest rate", min_value=0.0, value=0.01)
sigma = st.sidebar.number_input("Volatility", min_value=0.0, value=0.2)
mu = st.sidebar.number_input("Expected return", min_value=0.0, value=0.05)
dt = 1/252

if st.sidebar.button("Generate"):
    t, stock_prices = Util.simulate_stock_prices(S0, mu, sigma, T, dt)
    st.line_chart(stock_prices)

st.sidebar.write("---") 



if st.sidebar.button("Back Testing"):
    st.write("Back Testing started...")




