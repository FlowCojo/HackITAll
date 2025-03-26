from openai import OpenAI
from dotenv import load_dotenv
import os
import time
import json

load_dotenv('../.env')
client = OpenAI()
client.api_key=os.getenv('OPENAI_API_KEY')

ASSITANT_RUN_WAITING_TIME = 0.5
def chat_with_assistant(prompt, thread_id):
    """
    Send a message to the OpenAI assistant and return the response
    """

    try:
        if not thread_id: 
            thread = client.beta.threads.create()  # Create a chat thread if not already one
            thread_id = thread.id

        client.beta.threads.messages.create(thread_id=thread_id, role="user", content=prompt)

        create_run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=os.getenv('OPENAI_ASSISTANT_ID')
        )

        time.sleep(ASSITANT_RUN_WAITING_TIME)

        run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=create_run.id)

        tools_to_submit = []
        while run.status != 'completed':
            
            if run.status == 'requires_action':
                tool_outputs = []
                
                for tool in run.required_action.submit_tool_outputs.tool_calls:
                    # Submit output to the assistant
                    tool_data = {
                        "tool_call_id": tool.id,
                        "output": "{'success': true}"

                    }

                if run.required_action.type == 'submit_tool_outputs':
                    client.beta.threads.runs.submit_tool_outputs(thread_id=thread_id, run_id=run.id, tool_outputs=tool_outputs)  
                                        
            if run.status in ['failed', 'incomplete', 'expired']:
                # Handle other statuses: 'failed', 'incomplete', 'expired'

                print(f"Assistant Run {run.status.capitalize()}!")
                return None

            time.sleep(ASSITANT_RUN_WAITING_TIME)
            run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=create_run.id)


        # Retrive response
        response = client.beta.threads.messages.list(thread_id)
        
        return response.data[0].content[0].text.value, thread_id

    except Exception as e:
        print(f"There was a problem calling the OpenAI Assistant: {e}")
        return None, thread_id
    
