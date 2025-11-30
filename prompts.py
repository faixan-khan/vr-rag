prompt = {}
prompt['one'] = 'I have an image of a bird and need to identify its specie type. I have narrowed it down to five possible species. Please use the following descriptions to determine the most likely specie: {}. \n Analyze the bird in the image, considering its physical characteristics and compare them to the given specie descriptions. Provide only the name of the most likely specie.'
prompt['two'] = 'I have an image of a bird and need to identify its specie type. I have narrowed it down to five possible species. Please use the following descriptions to determine the most likely specie: {}. \n Analyze the bird in the image, considering its physical characteristics and compare them to the given specie descriptions. Provide only the name of the most likely specie. Ensure the name is from the descriptions.'
prompt['three'] = 'Analyze the provided bird image in conjunction with the following 5 species descriptions. Your task is to identify which of the five species the bird in the image most accurately belongs to. The selected species must be one of the five provided options. Provided Species Descriptions: {}. Based on your visual analysis of the image and careful consideration of the provided descriptions, state the name of the species depicted in the image.'
prompt['gemini'] = 'Analyze the provided bird image in conjunction with the following 5 species descriptions. Your task is to identify which of the five species the bird in the image most accurately belongs to. The selected species must be one of the five provided options, and your decision should be based solely on the visual characteristics present in the image compared to the descriptions. \
Provided Species Descriptions: {} \
State only the common name of the species depicted in the image, exactly as it appears in the provided descriptions.'
prompt['gpt'] = "You are given an image of a bird and five possible species it may belong to. \
Below are the species descriptions: {}. \
Analyze the bird in the image and compare it with the descriptions. \
Output only the name of the most likely species, choosing strictly from the five provided."


prompt['gemini_fish'] = 'Analyze the provided fish image in conjunction with the following 5 species descriptions. Your task is to identify which of the five species the fish in the image most accurately belongs to. The selected species must be one of the five provided options, and your decision should be based solely on the visual characteristics present in the image compared to the descriptions. \
Provided Species Descriptions: {} \
State only the common name of the species depicted in the image, exactly as it appears in the provided descriptions.'
prompt['gemini_pok'] = 'Analyze the provided Pokémon image in conjunction with the following 5 species descriptions. Your task is to identify which of the five species the Pokémon in the image most accurately belongs to. The selected species must be one of the five provided options, and your decision should be based solely on the visual characteristics present in the image compared to the descriptions. \
Provided Species Descriptions: {} \
State only the common name of the species depicted in the image, exactly as it appears in the provided descriptions.'


def get_prompt(prompt_type):
    return prompt[prompt_type]