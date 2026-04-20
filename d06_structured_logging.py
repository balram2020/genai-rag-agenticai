# Logging module in Python
import logging
import json 
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import uuid

load_dotenv()

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger("llm_agent")

llm_openai = ChatOpenAI(model="gpt-4.1-nano")
conversation = [SystemMessage(content="You are a concise, professional, and friendly assistant")]

def logged_invoke(user_message: str, session_id: str = "demo_session"):
    start_time = datetime.now()
    conversation.append(HumanMessage(content=user_message))
    logger.info(json.dumps({
        "event": "llm_call_start",
        "session_id": session_id,
        "timestamp": start_time.isoformat(),
        "user_message": user_message,
        "model": "gpt-4.1-nano",
    }))
    
    try:
        response = llm_openai.invoke(conversation)
        latency = (datetime.now() - start_time).total_seconds()
        print(f"Latency: {latency}ms")
        logger.info(json.dumps({
            "event": "llm_call_success",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "latency": latency,
            "response": response.content,
            "output_tokens": response.usage_metadata.get("output_tokens"),
            "input_tokens": response.usage_metadata.get("input_tokens"),
            "output_token_cost": response.usage_metadata.get("output_tokens")* 0.0000015, # we get the actual cost from the API and load it from the .env file / config file
            "input_token_cost": response.usage_metadata.get("input_tokens")* 0.00000015, # we get the actual cost from the API and load it from the .env file / config file
            "total_cost": response.usage_metadata.get("output_tokens")* 0.0000015 + response.usage_metadata.get("input_tokens")* 0.00000015, # we get the actual cost from the API and load it from the .env file / config file
        }))
        return response.content
    except Exception as e:
        latency = (datetime.now() - start_time).total_seconds()
        logger.error(json.dumps({
            "event": "llm_call_error",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "latency": latency,
            "error": str(e),
        }))
        raise e

session_id = str(uuid.uuid4())
result = logged_invoke("What is the capital of France?", session_id=session_id)
print(result)



