import openai

from .media import encode_image

STAGE_1_PROMPT = """
Analyze the provided image. Your task is to identify all the main objects in the image and return a structured dictionary. 
Each key should be the name of the whole object, e.g., "a bear", instead of "a head" or "a paw".
The objects may be abstract outlines, shapes, etc. But you should avoid using descriptive adjectives, such as "abstract", "skeletal", or "outline", in the object name. You can make bold guesses, such as guessing that a grey cylindrical shape in the image is a chimney.
Each value should be a dictionary with the following keys:
1. "visible part": the part of the object that is visible in the image, such as "full body", "upper body", "head", or "leg". Avoid descriptive adjectives.
2. "shooting angle": from which direction the object is viewed, such as "front shot", "low-angle side shot", or "bird's eye view".
Use this format for your output:
{
  "object(s)": {
    "visible part": "...",
    "shooting angle": "...",
  }
}
For example, if the image shows the head of a bear viewed from the front and centered in the frame, the output should be:
{
  "a bear": {
    "visible part": "head",
    "shooting angle": "front shot",
  }
}
The output should be one dictionary only, without any additional sentences, explanations, or markdown syntax (json). 

"""

STAGE_2_PROMPT = """
You will be given a dictionary and an input sentence. Your task is to modify the dictionary based on the sentence and output the modified dictionary. This includes two steps:

First, throughout the task, the objects you need to identify are physical objects rather than stylistic forms. For instance, the term "art" in “a piece of art” denotes a physical object, while "art" in “an art of a cat” refers to a stylistic form. For each object in the dictionary, look in the sentence for any semantically related object. Two objects should be considered semantically related if they are:
0.  The same object (e.g., "a cow" and "a cow", "a horse" and "a horse")
1.  Visually similar in shape, silhouette, or structure (e.g., "a horse" and "a giraffe", "an egg" and "an oval object")
2.  Functionally or categorically similar (e.g., "a cat" and "a lion", "fruits" and "vegetables")

Second, for each semantically matched object, replace the object in the dictionary with the corresponding object in the sentence. Then output the modified dictionary. 
For example, if the input sentence is “An image of a horse” and the dictionary is { “cow”: { “visiblepart”: “head”, "shooting angle": "side shot" }, “farmhouse”: { “visible part”: “full-body”, "shooting angle": "front shot" } }, you should first match "cow" in the dictionary with "horse" in the sentence, and then replace the "cow" with "horse". In this case, your output should be: {“horse”: {“visible part”: “head”, "shooting angle": "side shot" }, “farmhouse”: {“visible part”: “full-body”, "shooting angle": "front shot" }}.

Your output should be one dictionary only, without any additional sentences, explanations, or markdown syntax (json). If the input dictionary is empty or invalid, return an empty dictionary {}. 

"""

STAGE_3_PROMPT = """
You will be given a dictionary and an input sentence. Your task is to modify the sentence based on the dictionary and output the modified sentence. 
Specifically, for each object in the dictionary, if the object occurs in the sentence, modify the sentence by adding the "visible part" and "shooting angle" descriptions from the dictionary to the object in the sentence. Keep the original descriptions in the sentence intact.
For all the unmatched objects in the dictionary, append them at the end of the sentence using this format (you do not need to describe them here): "[list of unmatched objects, separated by commas] in the background".
Return only the final modified sentence, expressed in fluent, natural English.

As an example, if the input sentence is “An image of a horse” and the dictionary is { “horse”: { “visible part”: “head”, "shooting angle": "side shot", }, “farmhouse”: { “visible part”: “full-body”, "shooting angle": "front shot", } }, you should first match the "horse" in the dictionary with the "horse" in the sentence, and then modify the sentence. Your output should be similar to “A side shot image of a horse's head. A farmhouse in the background.”

Your output should be the sentence only, without any additional sentences, explanations, or markdown syntax (json). If the input dictionary is empty or invalid, return the input sentence unchanged.

"""

def compose_prompt(client_type, api_key, base_url, model_name, raw_prompt, image_path):
    assert client_type == "openai", "Only OpenAI client is supported for now."

    client = openai.OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    img_str = f"data:image/jpeg;base64,{encode_image(image_path)}"
    
    response_1 = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": STAGE_1_PROMPT},
                    {"type": "image_url", "image_url": {"url": img_str}}
                ],
            }
        ],
    )

    response_1_content = response_1.choices[0].message.content

    print("response_1_content: ", response_1_content)

    response_2 = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": STAGE_2_PROMPT + '\n' + 'Dictionary: ' + response_1_content + '\n' + 'Input sentence: ' + raw_prompt},
                ],
            }
        ],
    )

    response_2_content = response_2.choices[0].message.content

    print("response_2_content: ", response_2_content)

    response_3 = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": STAGE_3_PROMPT + "Dictionary: " + response_2_content + "\n" + "Input sentence: " + raw_prompt},
                ],
            }
        ],
    )

    response_3_content = response_3.choices[0].message.content

    print("response_3_content: ", response_3_content)

    return response_3_content