from openai import OpenAI


def as_dict(**kwargs):
    return kwargs


def get_completion(conv):
    # see https://github.com/meta-llama/llama-recipes/blob/main/recipes/quickstart/Prompt_Engineering_with_Llama_3.ipynb
    extra_args = as_dict(
        model="tgi",
        # max_tokens=out_num_tokens,
        temperature=0.6,
        top_p=0.9,
    )
    if isinstance(conv, str):
        completion = client.completions.create(
            prompt=conv,
            **extra_args,
        )
        assert len(completion.choices) == 1, f"got {completion.choices}"
        return completion.choices[0].text
    else:
        completion = client.chat.completions.create(
            messages=conv,  # not conv[:-1]
            **extra_args,
        )
        assert len(completion.choices) == 1, f"got {completion.choices}"
        return completion.choices[0].message.content


# %%
endpoint = "http://i205:3000/v1"
# endpoint = "http://localhost:3000/v1"
client = OpenAI(
    base_url=endpoint,
    api_key="_",
)
math_problem = 'ABCD is a right angle trapezoid, with AB perpendicular to AD. Let E be a point such that line CD is' \
              ' the bisector of ∠ACE. I is the excenter of triangle DAB with touchpoints F, G, H. J is a points ' \
              'such that J is the point of concurrence of the exterior angle bisectors of triangle AIB. Define point' \
              ' K such that line KB and line HD are parallel. line FK and line AG are parallel. line KB intersects ' \
              'line FK at K.'
# math_problem = 'Let A, B, D, C be points such that trapezoid ABCD is a trapezoid where line AD is equal to line BC. ' \
#                'Let E be a point such that E is on the circle centered at B with radius BD'

prompt = 'imagine this is the description of a geometry problem. \n ' \
         f'{math_problem} \n can you rephrase it such that the ' \
         'meaning is preserved but it is said in a different way. Imagine exactly how a text book of geometry would ' \
         'write the problem. Pay special attention to the points and lines and their naming. Give me only the ' \
         'rephrased version. Please do not write anything extra.'
print(f'Prompt sent: \n {prompt}')
response = client.completions.create(
    model="tgi",
    prompt=prompt,
    max_tokens=500,
)

verify_prompt = f'Is the following math problem same as its rephrasing? If not give a correct rephrasing. ' \
                f'Just the rephrasing nothing else. \nOriginal: {math_problem}' \
                f'\nRephrased: {response.to_dict()["choices"][0]["text"]}'

print(f'Verify Prompt sent: \n {verify_prompt}')
response = client.completions.create(
    model="tgi",
    prompt=verify_prompt,
    max_tokens=500,
)

print("\n\n\n\n", response.to_dict()['choices'][0]['text'])
