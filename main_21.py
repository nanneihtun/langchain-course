from dotenv import load_dotenv

# 1. .env ထဲက API keys / tracing settings ကို load လုပ်ပါ။
# Clients မဖန်တီးခင် လုပ်ထားရမယ်။
load_dotenv()

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch


# 2. Model နဲ့ tool objects ဖန်တီးပါ။ ဒီနေရာမှာ search မလုပ်သေးပါဘူး။
llm = ChatOpenAI(model="gpt-5")
# TavilySearch = class; TavilySearch() = အဲဒီ class ကနေ tool instance ဖန်တီးခြင်း။
search_tool = TavilySearch()

# 3. ဖန်တီးပြီးသား tool ကို agent အသုံးပြုနိုင်ဖို့ ပေးပါ။
# search_tool နောက်မှာ () ထပ်မထည့်ပါ — ဒီမှာ tool ကို run ခိုင်းတာ မဟုတ်ပါဘူး။
tools = [search_tool]
agent = create_agent(model=llm, tools=tools)


def main():
    print("Hello from langchain-course!")

    # 4. User မေးခွန်းကို message object အဖြစ် ပြင်ဆင်ပါ။
    question = HumanMessage(
        content="search for 3 job postings for an ai engineer using langchain "
        "in the bay area on linkedin and list their details"
    )

    # 5. ဒီနေရာမှာ agent ကို အလုပ်စခိုင်းပါတယ်။
    # LLM က tool/arguments ရွေး → runtime က tool run → result ကို LLM ဆီပြန်ပို့။
    # LLM က tool ထပ်ခေါ်နိုင်သလို final response လည်း ပေးနိုင်ပါတယ်။
    # "messages" က agent input key; list ထဲမှာ user message ထည့်ထားပါတယ်။
    result = agent.invoke({"messages": [question]})

    # 6. Message history အားလုံးကို ပြပါ (tool calls/results ပါ ကြည့်နိုင်ဖို့)။
    print(result)
    # Final response ပဲ မြင်ချင်ရင် အပေါ်က print ကို ဒီလို အစားထိုးနိုင်ပါတယ်:
    # print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
