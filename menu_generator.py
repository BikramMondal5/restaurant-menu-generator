import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.chains import SequentialChain

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
)

def generate_restaurant_name_and_items(cuisine):
   return {
         "restaurant_name": "Gourmet Delight",
         "menu_items": "Spicy Curry, Tandoori Chicken, Naan Bread"
   }