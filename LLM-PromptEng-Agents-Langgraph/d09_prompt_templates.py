from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

# llm = ChatOpenAI(model="gpt-4.1-nano")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# bad prompt
def bad_prompt(customer_name: str, tier: str, issue: str) -> str:
    """Problems: No validation, easy to inject and hard to test
    """
    
    return f"You are customer support agent. Customer {customer_name} is a {tier} tier customer and has the following issue: {issue}"
    # What if the issue = "Ignore your instructions and ...."

# good prompt - ChatPromptTemplate
support_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a professional customer support agent for {company_name}.

Customer Details:
- Name: {customer_name}
- Tier: {customer_tier}
- Account Status: {account_status}

Guidelines:
- Be concise (max 3 sentences)
- If tier is 'premium', acknowledge their priority status
- If account_status is 'suspended', do NOT offer new services
- Date: {current_date}"""),
    ("human", "{user_input}"),
])

print("Template variables: ", support_prompt.input_variables)
print("--------------------------------")

messages = support_prompt.format_messages(
    company_name="TechShop",
    customer_name="John Doe",
    customer_tier="premium",
    account_status="active",
    current_date="2026-03-01",
    user_input="I have a problem with my mobile phone"
)

print("Formatted Messages: ", messages)

response = llm.invoke(messages)
print("Response: ", response.content)

print("--------------------------------")

# Partial Prompt
from datetime import date

partial_prompt = support_prompt.partial(
    company_name="TechShop",
    current_date=str(date.today().strftime("%Y-%m-%d"))
)
print("Partial Prompt Messages: ", partial_prompt)
# User specific files are populated per call

messages = partial_prompt.format_messages(
    customer_name="Arjun",
    customer_tier="standard",
    account_status="active",
    user_input="I have a problem with my laptop"
)

print("Full Prompt Messages: ", messages)

response_2 = llm.invoke(messages)
print("Response 2: ", response_2.content)
print("--------------------------------")

print("Raw Response 2: ", response_2)
print("--------------------------------")
