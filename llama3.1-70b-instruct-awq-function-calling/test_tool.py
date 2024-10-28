import os
from openai import OpenAI

user_message = os.environ.get("MESSAGE")

if user_message is None:
    user_message = "What is the weather like for the coming 3 days in Glasgow, Scotland? Select a function"

model = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
base_url = "http://localhost:3000/v1"
client = OpenAI(base_url=base_url, api_key='na')

messages = [
    {
        "role": "system",
        "content": "Do not make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."
    },
    {
        "role": "user",
        "content": user_message,
    }
]

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_n_day_weather_forecast",
            "description": "Get an N-day weather forecast",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "format": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use. Infer this from the users location."
                    },
                    "num_days": {
                        "type": "integer",
                        "description": "The number of days to forecast"
                    }
                },
                "required": ["location", "format", "num_days"],
                "additionalProperties": False,
            }
        }
    }
]

tool_choice = "auto"

response = client.chat.completions.create(
    model=model,
    messages=messages,
    tools=tools,
    tool_choice=tool_choice,
)

print(response)
