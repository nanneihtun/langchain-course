from pydantic import BaseModel, Field
from dotenv import load_dotenv

# 1. .env ထဲက API keys / tracing settings ကို load လုပ်ပါ။
# Clients မဖန်တီးခင် လုပ်ထားရမယ်။
load_dotenv()

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

# 2. Agent ပြန်ပေးရမယ့် data ပုံစံ (schema) ကို သတ်မှတ်ပါ။
# ဒီ classes တွေက search မလုပ်ပါ — answer နဲ့ sources ရဲ့ ပုံစံကို သတ်မှတ်တာပါ။
class Source(BaseModel):
    """Schema for a source used by the agent"""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer: str = Field(description="The agent's answer to the query")
    # list[Source] = Source objects ပါတဲ့ list။
    # default_factory=list = sources မပေးထားရင် object တစ်ခုစီအတွက် empty list အသစ်ဖန်တီးမယ်။
    sources: list[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer"
    )


# 3. Model နဲ့ tool objects ဖန်တီးပါ။ ဒီနေရာမှာ search မလုပ်သေးပါဘူး။
llm = ChatOpenAI(model="gpt-5")
# TavilySearch = class; TavilySearch() = အဲဒီ class ကနေ tool instance ဖန်တီးခြင်း။
search_tool = TavilySearch()

# 4. ဖန်တီးပြီးသား tool ကို agent အသုံးပြုနိုင်ဖို့ ပေးပါ။
# search_tool နောက်မှာ () ထပ်မထည့်ပါ — ဒီမှာ tool ကို run ခိုင်းတာ မဟုတ်ပါဘူး။
tools = [search_tool]
# response_format က schema class ကို လက်ခံတာမို့ AgentResponse နောက်မှာ () မထည့်ပါ။
# Agent က နောက်ဆုံးအဖြေကို ဒီ schema နဲ့ ကိုက်ညီအောင် ပြန်ပေးရမယ်။
# Schema ကိုက်ညီခြင်းက job posting တကယ် active ဖြစ်ကြောင်း အာမခံတာတော့ မဟုတ်ပါ။
agent = create_agent(
    model=llm,
    tools=tools,
    response_format=AgentResponse,
)


def main():
    print("Hello from langchain-course!")

    # 5. User မေးခွန်းကို message object အဖြစ် ပြင်ဆင်ပါ။
    question = HumanMessage(
        content="search for 3 job postings for an ai engineer using langchain "
        "in the bay area on linkedin and list their details"
    )

    # 6. ဒီနေရာမှာ agent ကို အလုပ်စခိုင်းပါတယ်။
    # LLM က tool/arguments ရွေး → runtime က tool run → result ကို LLM ဆီပြန်ပို့။
    # LLM က tool ထပ်ခေါ်နိုင်သလို final response လည်း ပေးနိုင်ပါတယ်။
    # "messages" က agent input key; list ထဲမှာ user message ထည့်ထားပါတယ်။
    result = agent.invoke({"messages": [question]})

    # 7. result က dictionary ဖြစ်ပြီး အဓိက data နှစ်ခု ပါပါတယ်။
    # result["messages"] = user/model/tool message history။
    # result["structured_response"] = schema နဲ့ စစ်ဆေးပြီးသား AgentResponse object။
    response = result["structured_response"]

    # Dictionary key ကို ["..."] နဲ့ယူပြီး object ရဲ့ field ကို .answer / .sources နဲ့ယူပါ။
    print(response.answer)
    print("\nSources:")
    for source in response.sources:
        print(f"- {source.url}")

    # Debugging အတွက် history အပါအဝင် result အားလုံး မြင်ချင်ရင် ဒီလိုင်းကို ဖွင့်ပါ။
    # print(result)


if __name__ == "__main__":
    main()
