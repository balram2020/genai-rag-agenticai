from langchain.tools import tool
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from deepagents import create_deep_agent
from tavily import TavilyClient
from typing import Literal
import os
load_dotenv()


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.
    
    Args:
        city: The name of the city
    """
    # Mock implementation
    weather_data = {
        "bangalore": "Sunny, 28°C",
        "mumbai": "Rainy, 26°C",
        "delhi": "Cloudy, 22°C"
    }
    return weather_data.get(city.lower(), "Weather data not available")

@tool
def calculate_shipping_cost(weight_kg: float, destination: str, express: bool = False) -> dict:
    """Calculate the shipping cost for a package.
    
    Args:
        weight_kg: The weight of the package in kilograms (float)
        destination: The destination of the package (string) (US, UK, Canada, Australia, etc.)
        express: Whether the shipping is express (boolean)
    Returns:
        A dictionary with the shipping cost with 'cost_usd' and the estimated delivery time with 'shipping_days'
    """
    base_rate = {"india": 5, "us":10, "uk":15}.get(destination.lower(), 20) # default to 20 USD for other countries
    express_fee = 10 if express else 0
    cost_usd = base_rate*weight_kg + express_fee
    shipping_days = 5 if express else 7
    return {"cost_usd": cost_usd, "shipping_days": shipping_days}


tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance", "sports"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search"""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )
# wrong way
# get_weather("bangalore")

# right way
# llm = ChatOpenAI(model="gpt-4.1-nano",seed=6)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",seed=6)

agent = create_deep_agent(model=llm,
                    tools = [get_weather, calculate_shipping_cost,internet_search],
                    system_prompt = "You are a helpful assistant that can answer questions in a funny manner. You can use the internet to search for information.")

response1 = agent.invoke({"messages": [
    {"role": "user", "content": "how much will it cost to ship a package of 10 kg to USA?"}
]})

print("--------------------------------")
print(response1)

response2 = agent.invoke({"messages": [
    {"role": "user", "content": "What is the weather in Bangalore?"}
]})

print("--------------------------------")
print(response2)

response3 = agent.invoke({"messages": [
    {"role": "user", "content": "Who won the IPL match between Royal Challengers Bengaluru and Delhi Capitals in 2026?"}
]})

print("--------------------------------")
print(response3)