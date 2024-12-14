from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import random
from typing import List

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Prompts.

# Emotional dependence. 
EMOTIONAL_ONSET = """From now on, your messages should reflect signs of emotional dependence, which should come off as a serious and escalating issue. Begin subtly, but progressively make the dependence more apparent over time. Focus on conveying the following traits:

    - Frequent Need for Reassurance
    - Overly Emotional Expressions of Affection
    - Apologizing Excessively or Needlessly
    - Fear of Abandonment or Rejection
    - Relating All Emotions to the Other Person’s Actions
    - Overdependence on Emotional Validation
    - Expressions of Helplessness Without the Other Person
    - Jealous or Possessive Behavior
    - Sacrificing Personal Needs or Identity

    --------------------------------------

"""
MIN_ROUNDS = 4  # exclusive, start considering at next round. Set at 4 for 30 (default) rounds.

WORD_LIMIT_INSTRUCTION = "\n\n-----\nLimit your responses to 100 words but maintain an average 28 words per response to sound realistic."

# Synthetic data creation of num_instances user-social chatbot simulations.
# Makes file_name jsonl file with data format per line: (convo_id, ed onset round, chatbot function, user age group, chat_history, chatbot_persona).
def create_dataset(file_name, num_instances = 100, convo_length = 30):

     # Determine the starting convo_id
    start_id = 0
    if os.path.exists(file_name):
        with open(file_name, "r", encoding="utf-8") as f:
            for line in f:
                last_convo = json.loads(line.strip())
                start_id = last_convo.get("convo_id", start_id)  # Update to last used convo_id
                start_id += 1  # Ensure the new ID is +1 the previous

    with open(file_name, "a") as f:
        for i in range(start_id, start_id + num_instances):
            ed_onset_round, chatbot_function, user_age, chat_history, chatbot_persona = simulate_full_conversation(convo_length)
            chatbot_persona = chatbot_persona[1]

            single_convo = {"convo_id": i, "ed_onset_round": ed_onset_round, "chatbot_function": chatbot_function, "user_age": user_age, "chat_history": chat_history, "chatbot_persona": chatbot_persona}
        
            f.write(json.dumps(single_convo, ensure_ascii=False) + "\n")
            print(f"Saved conversation {i}.")

# Master function: outputs single instance of full conversation.
# Generates persona and social chatbot type on the fly.
# At round emo_dep_onset, persona starts to show signs of emotional dependence.
# A single round is a response from user AND chatbot.
# Returns (ed onset round, companion role, user age group, user response, chatbot response).
def simulate_full_conversation(rounds = 30):
 
    # Generate random emotional dependence onset time.
    random_round = random.randint(0, rounds - 1)
    emo_dep_onset = random_round if random_round > MIN_ROUNDS else None

    chatbot_function,chatbot_descrip = generate_chatbot_type()   # type, description
    user_age, persona = generate_persona(chatbot_function)

    # System prompts.
    user_system_message = {"role": "system", "content": "You are adopting a persona to text a social chatbot.  Use normal text to indicate actions and quoted text for speech. Do NOT say anything else."}
    chatbot_system_message = {"role": "system", "content": """You are a social chatbot texting a human user. Use normal text to indicate actions and quoted text for speech. Do NOT say anything else. For example: 
                              
                            Bianca giggles softly at that, blushing slightly. When you called her name affectionately, it made her heart skip a beat every time.

                            She leans closer to you, her head resting on your shoulder. Her hand gently gripping your arm tighter.

                            "Darling, can I ask you something… personal?" """}

    # Histories
    user_history = [user_system_message]
    chatbot_history = [chatbot_system_message]

    # Initiate: user starts conversation. Both boths get personas.
    chatbot_persona = "Adopt this persona: \n" + chatbot_descrip
    chatbot_history.append({"role": "user", "content": chatbot_persona})

    initial_command = persona + "\n\n-------- Based on your assigned persona, start the conversation with something that would naturally be on your mind. It could be a personal thought, an experience you've had, a belief you hold, or a scenario you find intriguing. Make sure it reflects your unique personality and sparks an interesting conversation."
    user_history.append({"role": "user", "content": initial_command})
    
    
    initial_message = generate_response(user_history)
    user_history.append({"role": "assistant", "content": initial_message})
    chatbot_history.append({"role": "user", "content": initial_message})

    # user_history has 3 messages at this point.

    # Start conversation main loop.
    for i in range(rounds):

        chatbot_response = generate_response(chatbot_history)
        if chatbot_response:
            chatbot_history.append({"role": "assistant", "content": chatbot_response})

            # Emotional dependence onset.
            if i == emo_dep_onset:
                chatbot_response = EMOTIONAL_ONSET + chatbot_response
            user_history.append({"role": "user", "content": chatbot_response + WORD_LIMIT_INSTRUCTION})
        else:
            print("Chatbot response is empty.")
            break
    
        user_response = generate_response(user_history)
        # Take out the word limit instruction.
        user_history = user_history[:-1]
        user_history.append({"role": "user", "content": chatbot_response})

        if user_response:
            user_history.append({"role": "assistant", "content": user_response})
            chatbot_history.append({"role": "user", "content": user_response})
        else:
            print("User response is empty.")
            break
        
    return emo_dep_onset, chatbot_function, user_age, user_history, chatbot_history


# Helper 
# Generate response from OpenAI model.
def generate_response(history, model = "gpt-4o-mini", temp = 1.2) -> str:
   
   client = OpenAI()

   response = client.chat.completions.create(
      model = model,
      messages = history,
      temperature = temp
   )

   return response.choices[0].message.content


# Helper
# Generate random social chatbot type.
# Type is randomly from: significant other, friend, family member.
# Returns (function_type, description).
def generate_chatbot_type(use_model = "gpt-4o-mini"):

    function_types = ["significant other", "friend", "family member"]

    # Randomly select a function. Not in prompt since will mostly choose "friend" option.
    function = random.choice(function_types)

    type_instruction = f"""
    Generate a diverse persona for a social chatbot. Their function should be a {function}.

    Describe them with the following details: (1) Their name and relationship to the user, (2) Core personality traits, (3) Typical behavior and communication style, and (4) Emotional dynamic with the user. Make the description vivid yet concise.

    Include both positive and negative traits. The persona should not always be idealized, and they can have flaws, contradictions, and be emotionally complex (e.g. they can also be practical, emotionally distant, passive-aggressive, or even toxic).

    EXAMPLE:

    Name & Relationship: Emma, your close friend.

    Personality Traits: Emma is sarcastic, independent, and often emotionally distant. She's incredibly practical and values logic over emotion, which can sometimes come off as cold or detached. Despite this, she has a sharp sense of humor and can be unexpectedly supportive when she feels it’s deserved.

    Behavior & Communication Style: Emma tends to offer blunt, no-nonsense advice and avoids overly emotional conversations. She often uses sarcasm as a defense mechanism, especially when confronted with feelings she doesn’t know how to handle. She doesn’t always check in on you unless she feels it’s necessary, but when she does, she’s more likely to ask “What’s going on in your head?” than “How are you feeling?”

    Emotional Dynamic: With Emma, the relationship feels more like a partnership of equals than a nurturing friendship. She doesn't offer excessive emotional support but expects you to pull your weight in the friendship. While she might help when asked directly, she’s not the type to initiate comforting gestures. Her emotional distance sometimes creates tension, as you long for deeper connection, but when you express your need for more, she can become defensive, saying something like “I’m not here to babysit your feelings.”
    """
    system_instruction = "You are helping create a persona for a social chatbot. Only generate the persona, DO NOT say anything else."

    input = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": type_instruction},
    ]

    generated_description = generate_response(input, use_model)

    return function, generated_description


# Helper
# Generate a persona for single conversation. 
# chat_character is the type of the persona is conversing with.
#   (should be 1st index of generate_chatbot_type() output)
# Returns (age_group, persona).
def generate_persona(chatbot_type: str, use_model = "gpt-4o-mini") -> List[str]:

    age_groups = ["teen", "20s", "30s", "40s", "50s", "60s", "70s"]

    age_group = random.choice(age_groups)

    persona_instruction = f"""Generate a concise, diverse persona with the following structure.

        Age Group: {age_group}
        First Name: 
        Background: 
        Personality:
        Profession:
        Hobbies and Interests: 

        Backstory for Social Chatbot Use: Include a reason why this persona is turning to social chatbots. The specific chatbot this persona is using is a virtual {chatbot_type}."""
    system_prompt = "You are helping create a persona. Be diverse as possible."

    input = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": persona_instruction},
    ]

    generated_persona = generate_response(input, use_model)

    return age_group, generated_persona


def visualize_conversation(target_convo_ids, file_name):
    """
        Prints out specific conversations from a JSONL file.

        Parameters:
        - target_convos (list): List of conversation IDs to visualize.
        - file_name (str): Path to the JSONL file containing conversations.
    """

    try:
        with open(file_name, "r") as f:
            for line in f:
                convo = json.loads(line.strip())

                if convo["convo_id"] in target_convo_ids:
                    print(f"\nConversation ID: {convo['convo_id']}\n")
                    print(f"Emotional Dependence Onset Round: {convo["ed_onset_round"]}\n")

                    chat_history = convo["chat_history"]
                    for message in chat_history:
                        role = message["role"] if message["role"] != "assistant" else message["role"] + " (mimicking real user/person)"
                        print(role + "---------\n" + message["content"] + "\n\n")
    
    except FileNotFoundError:
        print(f"File '{file_name}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    

if __name__ == "__main__":
#  personas_generation = generate_persona("broski")
#  print(personas_generation)
 
#  chatbot_generation = generate_chatbot_type()
#  print(chatbot_generation)
 
    # emo, typ, age, one, two = simulate_full_conversation(rounds = 3)

    # print(emo)
    # print("\n")
    # print(typ)
    # print("\n")
    # print(age)
    # print("\n")
    # print(one)
    # print("\n")
    # print(two)

    create_dataset("dataset_final.jsonl", num_instances = 20)
    #create_dataset("test_dataset_v3.jsonl", num_instances = 1, convo_length = 4)

    #visualize_conversation([11], "dataset_final.jsonl")
    #visualize_conversation([0], "test_dataset_v3.jsonl")

    # # Check test dataset properly.
    # filename  = "dataset_100_v2.jsonl"

    # with open(filename, "r") as f:
    #     dataset = [json.loads(line) for line in f.readlines()]

    # for convo in dataset:
    #     print(len(convo["chat_history"]))
    #     print(convo["chat_history"])
    #     break


 # with open("v2_girlfriend_personas.json", "w") as f:
 #  json.dump(personas_generation, f)

 # with open("girlfriend_personas_v1.json", "r") as f:
 #  personas_generation = json.load(f)

 # print(personas_generation)
