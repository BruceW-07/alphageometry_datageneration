from random import random

import random
from openai import AzureOpenAI


class ChatHistManager:
    def __init__(self, api_key, api_version, azure_endpoint, model_name, max_tokens, is_dummy=False):
        self._chats = []
        if not is_dummy:
            self.client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=azure_endpoint,
            )
        else:
            self.first_prob_formalization_call = True
        self.is_dummy = is_dummy
        self.max_tokens = max_tokens
        self.model_name = model_name

    def get_dummy_resp(self, prompt):
        fl_prob = ''
        if 'You formalized' in prompt:  # Simplified string search
            fl_prob = (
                '- problem:\n'
                '    description: "C is a point on the perpendicular bisector of AB. Prove that AC = BC."\n'
                '    usage: "isosceles_triangle C A B"\n'
                '    constraints:\n'
                '      - "free D"\n'
                '      - "perp C D A B"\n'
                '    goal:\n'
                '      - "cong A C B C"\n'
                '    comment_templates:\n'
                '      - "The line segment {arg1}{arg0} is same length as {arg2}{arg0} id {arg2} lies on the perpendicular bisector of {arg1}{arg2}."\n'

            )

        elif 'I have a few predefined' in prompt:  # Simplified string search
            if self.first_prob_formalization_call:
                self.first_prob_formalization_call = False
                fl_prob = (
                    '- problem:\n'
                    '    description: "C is a point on the perpendicular bisector of AB. Prove that AC = BC."\n'
                    '    usage: "isosceles_triangle C A B"\n'
                    '    constraints:\n'
                    '      - "free D"\n'
                    '      - "midpoint D A B"\n'
                    '      - "perp C D A B"\n'
                    '    goal:\n'
                    '      - "cong A C B C"\n'
                    '    comment_templates:\n'
                    '      - "The line segment {arg1}{arg0} is same length as {arg2}{arg0} id {arg2} lies on the perpendicular bisector of {arg1}{arg2}."\n'

                )
            else:
                fl_prob = (
                    '- problem:\n'
                    '    description: "C is midpoint of AB"\n'
                    '    usage: midpoint C A B\n'
                    '    constraints:\n'
                    '      - "free C"\n'
                    '      - "cong C A C B"\n'
                    '      - "coll C A B"\n'
                    '    comment_templates:\n'
                    '      - "{arg0} is midpoint of {arg1}{arg2}"\n'
                )
        elif 'Here are two description of' in prompt:
            fl_prob = random.choice(['Yes', 'No'])
        return fl_prob

    def call_llm(self, chat_id, prompt):
        if len(self._chats) > chat_id:
            old_chats = self._chats[chat_id]
            old_chats.append({"role": "user", "content": prompt})
            self._chats[chat_id] = old_chats
        else:
            raise ValueError(f'Invalid chat id{chat_id}. We have only {len(self._chats)} active chats')

        if self.is_dummy:
            agent_msg = self.get_dummy_resp(prompt)
        else:
            response = self.client.chat.completions.create(
                messages=self._chats[chat_id],
                model=self.model_name,
                max_tokens=self.max_tokens,
            )
            agent_msg = response.choices[0].message.content
        self._chats[chat_id].append({"role": "assistant", "content": agent_msg})
        # TODO: Remove this
        self.dump_all_chats_to_file('all_chats.yaml')
        input('Press any button to continue ... ')
        return agent_msg

    def start_new_chat(self):
        chat_id = len(self._chats)
        print(f'Starting new chat{chat_id}')
        new_chat = [{"role": "system", "content": "You are a helpful assistant."},]
        self._chats.append(new_chat)
        return chat_id

    def get_chat_hist(self, chat_id):
        return self._chats[chat_id]

    def dump_all_chats_to_file(self, output_file):
        yaml_formated_chat = self.get_all_chats_as_yaml()
        with open(output_file, 'w') as file:
            file.write(yaml_formated_chat)

    def get_all_chats_as_yaml(self):
        """
            Convert a list of conversations to YAML format with chat index and write to a file.

            Args:
                conversations (list of dict): List of conversations. Each conversation should be
                                              a dictionary containing 'system', 'user', and 'assistant' keys.

            Example:
                conversations = [
                    [{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "do something."},
                    {"role": "assistant", "content": "here is teh job"},
                    {"role": "user", "content": "looks good"}],

                    [{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "do something."},
                    {"role": "assistant", "content": "here is teh job"},
                    {"role": "user", "content": "looks good"}],
                ]
        """
        output = []

        for chat_number, conversation in enumerate(self._chats):
            formatted_output = f"chat{chat_number + 1}:\n"
            for message in conversation:
                role = message['role'].lower()
                content = message['content']

                if role == 'system':
                    formatted_output += f"    system: '{content}'\n"
                elif role == 'user':
                    formatted_output += f"    user: '{content}'\n"
                elif role == 'assistant':
                    formatted_output += f"    Assistant: '{content}'\n"

            output.append(formatted_output.rstrip())

        return "\n".join(output)
