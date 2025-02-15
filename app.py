import streamlit as st
import openai
import requests
import yfinance as yf
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from duckduckgo_search import DDGS
import datetime

# Fetch API keys from Streamlit secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
WEATHERAPI_KEY = st.secrets["WEATHERAPI_KEY"]

# Headers for requests
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "application/json",
    "Connection": "keep-alive"
}

# 1. WEATHER TOOL (Using WeatherAPI.com)
def get_weather(city):
    """Fetches current weather for a given city using WeatherAPI.com."""
    url = f"http://api.weatherapi.com/v1/current.json?key={WEATHERAPI_KEY}&q={city}&aqi=no"

    response = requests.get(url, headers=HEADERS)

    if response.status_code == 200:
        data = response.json()
        temp_c = data["current"]["temp_c"]
        condition = data["current"]["condition"]["text"]
        humidity = data["current"]["humidity"]
        wind_kph = data["current"]["wind_kph"]
        
        return f"🌍 **Weather in {city}**:\n🌡 Temperature: {temp_c}°C\n🌤 Condition: {condition}\n💧 Humidity: {humidity}%\n💨 Wind Speed: {wind_kph} kph"
    else:
        return f"❌ Failed to fetch weather data for {city}."


# 2. STOCK PRICE TOOL (Using Yahoo Finance)
def get_stock_price(ticker):
    """Fetches the latest stock price for a given ticker symbol (NYSE)."""
    try:
        stock = yf.Ticker(ticker)
        price = stock.history(period="1d")["Close"].iloc[-1]
        return f"📈 The latest stock price for {ticker} is **${price:.2f}**."
    except Exception as e:
        return f"❌ Error fetching stock price: {str(e)}"


# 3. WEB SEARCH TOOL (Using DuckDuckGo Search)
def web_search(query, max_results=5):
    """Performs a web search using DuckDuckGo and returns the top results."""
    try:
        results = DDGS().text(query, max_results=max_results)

        if not results:
            return "❌ No relevant results found."

        # Format the results nicely
        formatted_results = "\n\n".join(
            [f"🔹 {res['title']}\n🔗 {res['href']}\n📄 {res['body']}" for res in results]
        )

        return formatted_results
    except Exception as e:
        return f"❌ Error in web search: {str(e)}"


def default_query(query):
    """This tool is called if no other tool seems suitable for the job"""
    return "The query is not very clear, ask the user to further elaborate or use default language model to answer"


# Create LangChain Tools
weather_tool = Tool(name="Weather Tool", func=get_weather, description="Fetches current weather for any city.")
stock_tool = Tool(name="Stock Price Tool", func=get_stock_price, description="Fetches the latest stock price for any NYSE-listed company.")
web_tool = Tool(name="Web Search Tool", func=web_search, description="Performs a web search using DuckDuckGo.")
default_tool = Tool(name="Default Tool", func=default_query, description="This tool is used when no other tool seems feasible")

# Initialize LangChain Agent with GPT-4o
llm = ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY)

agent = initialize_agent(
    tools=[weather_tool, stock_tool, web_tool,default_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)


# STREAMLIT UI
st.title("🔹 AI Agent")
st.header("Ask about weather, stock prices or general queries.")

# User Input Box
user_input = st.text_area("💬 Ask me anything:", height=100)

# Button to trigger agent
if st.button("Run Agent"):
    with st.spinner("🤖 Thinking..."):
        # Capture verbose output
        from io import StringIO
        import sys

        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()

        # Run the agent
        # response = agent.run(user_input)
        system_message = f"""You are a helpful assistant. Today's date is {datetime.datetime.date(datetime.datetime.now())}. Use only when needed especially when doing web search.
        If you don't understand the query or you think it is unclear like Test etc, please ask user to further elaborate before answering."""
        response = agent.run(f" {system_message} Here's the user's query.: {user_input}")

        # Get verbose logs
        verbose_output = mystdout.getvalue()
        sys.stdout = old_stdout

        # Display response
        st.subheader("🔹 AI Response:")
        st.markdown(response)

        # Display verbose logs
        st.subheader("📜 Agent Reasoning & Logs:")
        st.text_area("Agent Debug Info:", verbose_output, height=300)
