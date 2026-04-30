from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()

# model_openai = ChatOpenAI(model="gpt-4.1-nano",seed=6)
model_google = ChatGoogleGenerativeAI(model="gemini-2.5-flash",seed=6)


m1 = "Who is the Prime Minister of India?"
m2 = "What is his age?"

# response_openai = model_openai.invoke(m1)
response_google = model_google.invoke(m1)

# print(f"OpenAI: {response_openai.content}")
print(f"Google: {response_google.content}")

# response_openai = model_openai.invoke(m2)
response_google = model_google.invoke(m2)

# print(f"OpenAI: {response_openai.content}")
print(f"Google: {response_google.content}")

print("--------------------------------")
# print("OpenAI: ", response_openai)
print("Google: ", response_google)
print("--------------------------------")
# print(f"Type of object: {type(response_openai)}")
print(f"Type of object: {type(response_google)}")