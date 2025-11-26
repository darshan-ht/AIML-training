# pip install -qU "langchain[openai]" to call the model

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    # City to temperature mapping
    city_temps = {
        "pune": 10,
        "mumbai": 12,
        "delhi": 16,
    }
    
    # Convert city to lowercase for case-insensitive lookup
    city_lower = city.lower()
    
    if city_lower in city_temps:
        temp = city_temps[city_lower]
        return f"The temperature in {city} is {temp}°C"
    else:
        return f"Weather data not available for {city}"

# Create OpenAI model with custom base URL
model = ChatOpenAI(
    model="gpt-3.5-turbo",  # or any model name you want to use
    base_url="http://localhost:1234/v1",  # Replace with your custom base URL
    api_key="not-needed",  # May not be needed for local servers
)

agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# Run the agent
response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the temprature in delhi"}]}
)

# Print the response
print("\nAgent Response:")
# Extract the last message content (the agent's response)
if "messages" in response and len(response["messages"]) > 0:
    last_message = response["messages"][-1]
    if hasattr(last_message, "content"):
        print(last_message.content)
    elif isinstance(last_message, dict) and "content" in last_message:
        print(last_message["content"])
    else:
        print(last_message)
else:
    print(response)
