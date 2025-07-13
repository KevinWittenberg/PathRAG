import os
from PathRAG import PathRAG, QueryParam
from PathRAG.llm import gpt_4o_mini_complete

WORKING_DIR = "C:/Users/Kevin/Documents/Work/SCP/AI/PathRAG/PathRAG"

from dotenv import load_dotenv

# API secret key handling
load_dotenv()

# Load the API key from the .env file
api_key = os.getenv("api_key")

# Check if the key was found
if api_key:
    print(f"The API Key is: {api_key}")
else:
    print("API_KEY not found. Make sure it's set in your .env file.")

os.environ["OPENAI_API_KEY"] = api_key # type: ignore
base_url="https://api.openai.com/v1"
os.environ["OPENAI_API_BASE"]=base_url


if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = PathRAG(
    working_dir=WORKING_DIR,
    llm_model_func=gpt_4o_mini_complete,  
)

data_file="./testdata1.txt"
question="Wie is de directeur van het SCP? En wat is de relatie tussen SCP en PBL?"
with open(data_file) as f:
    rag.insert(f.read())

print(rag.query(question, param=QueryParam(mode="hybrid")))

