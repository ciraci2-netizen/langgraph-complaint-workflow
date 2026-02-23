from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

print("KEY LOADED:", os.getenv("OPENAI_API_KEY"))

client = OpenAI()

response = client.responses.create(
    model="gpt-4o-mini",
    input="Hello"
)

print(response.output[0].content[0].text)