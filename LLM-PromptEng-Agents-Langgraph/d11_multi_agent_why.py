from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import re
from langchain.agents import create_agent
import random
load_dotenv()
# llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

@tool
def get_order_status(order_id: str) -> str:
    """ Get order status by order id """
    return f"Order {order_id} is in {random.choice(['pending', 'shipped', 'delivered'])} status"

@tool
def process_return(order_id: str) -> str:
    """ Process return by order id """
    return f"Return {order_id} is processed. Return ID is {random.randint(1000, 9999)}"

@tool
def check_payment_status(order_id: str) -> str:
    """ Check payment status by order id """
    return f"Payment {order_id} is {random.choice(['pending', 'completed', 'failed'])}"

@tool
def check_inventory(product_id: str) -> str:
    """ Check inventory by product id """
    return f"Inventory {product_id} is {random.choice(['in stock', 'out of stock'])}"

@tool
def issue_refund(order_id: str) -> str:
    """ Issue refund by order id """
    return f"Refund for order {order_id} is issued. Refund ID is REF#{random.randint(1000, 9999)}"

@tool
def create_bug_report(issue: str) -> str:
    """ Create bug report by issue """
    return f"Bug report for issue {issue} is created. Bug ID is JIRA#{random.randint(1000, 9999)}"

@tool
def create_feature_request(feature: str) -> str:
    """ Create feature request by feature """
    return f"Feature request for feature {feature} is created. Feature ID is JIRA#{random.randint(1000, 9999)}"

@tool
def upgrade_subscription(customer_id: str) -> str:
    """ Upgrade subscription by customer id """
    return f"Subscription for customer {customer_id} is upgraded. Subscription ID is SUB#{random.randint(1000, 9999)}"

@tool
def get_subscription_status(customer_id: str) -> str:
    """ Get subscription status by customer id """
    return f"Subscription for customer {customer_id} is {random.choice(['active', 'inactive', 'cancelled'])}"

all_tools = [get_order_status, process_return, check_payment_status, check_inventory, issue_refund, create_bug_report, create_feature_request, upgrade_subscription, get_subscription_status]

order_agent = create_agent(
    # model=ChatOpenAI(model="gpt-4.1-nano"),
    model=ChatGoogleGenerativeAI(model="gemini-2.5-flash"),
    tools=all_tools,
        system_prompt="You are a helpful assistant that can help with customer support"
    )

#test Queries
messages = ["What is the status of order 1234567890?", "Process return for order 1234567890", "Check payment status for order 1234567890", "Check inventory for product 1234567890", "Issue refund for order 1234567890", "Create bug report for issue 1234567890", "Create feature request for feature 1234567890", "Upgrade subscription for customer 1234567890", "Get subscription status for customer 1234567890"]

for m in messages:
    result = order_agent.invoke({"messages":[
        {"role": "user", "content": m}
    ]})
    print(result)
    print("--------------------------------")
