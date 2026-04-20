 # System Message - Set the Agent Personality
 # Human Message - User Query
 # AI Message - model's response

from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage, AIMessage, SystemMessage
load_dotenv()

# llm = ChatOpenAI(model="gpt-4.1-nano",seed=6)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",seed=6)

messages = [SystemMessage(content="You are a helpful assistant that can answer questions in a funny manner."), HumanMessage(content="Who is the Prime Minister of India?")]

response = llm.invoke(messages)

print(response.content)

# Append the response to the messages to include the context
messages.append(response)

query_2 = HumanMessage(content="What is his age?")
# Append the query to the messages to include the context
messages.append(query_2)
#invoke the model with the messages
response_2 = llm.invoke(messages)
print("--------------------------------")
print(response_2.content)