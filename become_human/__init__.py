import os
from dotenv import load_dotenv

load_dotenv()

if not os.path.exists("./data"):
    os.makedirs("./data")
if not os.path.exists("./config"):
    os.makedirs("./config")
