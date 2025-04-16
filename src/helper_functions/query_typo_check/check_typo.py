from langchain.prompts import ChatPromptTemplate
import json
from langchain_core.output_parsers import StrOutputParser


def llm_layer_to_check_typos(query,prompt_dict,country_list,filters_dict, model):
    """
    Uses a language model layer to detect and correct grammar or typographical errors
    based on a structured prompt.

    Args:
        prompt_dict (dict): A dictionary containing the system and user prompts. Must have the keys:
            - 'system_prompt': Instructions or role definition for the model.
            - 'human_prompt_1': The initial user input or context.
            - 'human_prompt_2': The input text to be checked for typos or grammar issues.
        model (object, optional): A language model instance capable of handling LangChain-style 
                                  prompt chaining. Defaults to the `model` variable in scope.

    Returns:
        str: The processed output from the language model, typically the corrected or reviewed text.
    """
    filters_list = json.dumps(list(filters_dict.keys())).replace(" Filters","")
    # Create a prompt using ChatPromptTemplate by combining system and user instructions
    prompt = ChatPromptTemplate([
        ('system', prompt_dict['system_prompt']),
        ('user', prompt_dict['human_prompt_1']),
        ('user',eval(prompt_dict['human_prompt_2'])),
        ('user',eval(prompt_dict['human_prompt_3'])),
        ('user', prompt_dict['human_prompt_4'])
    ])

    # Chain the prompt through the model and then parse the string output
    chain = prompt | model | StrOutputParser()
    
    # `query` should be the final input triggering the model – assuming it's defined elsewhere
    output = chain.invoke(query)

    return output