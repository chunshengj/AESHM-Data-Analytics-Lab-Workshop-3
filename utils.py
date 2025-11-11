import requests
import json
from dateutil import parser
import pandas as pd
from pprint import pprint as original_pprint
import os
import openai
import numpy as np


# Distance formulas. 
# In this ungraded lab, distance formulas will be implemented here. However, in future assignments, you will import functions from specialized libraries.
def cosine_similarity(v1, array_of_vectors):
    """
    Compute the cosine similarity between a vector and an array of vectors.
    
    Parameters:
    v1 (array-like): The first vector.
    array_of_vectors (array-like): An array of vectors or a single vector.

    Returns:
    list: A list of cosine similarities between v1 and each vector in array_of_vectors.
    """
    # Ensure that v1 is a numpy array
    v1 = np.array(v1)
    # Initialize a list to store similarities
    similarities = []
    
    # Check if array_of_vectors is a single vector
    if len(np.shape(array_of_vectors)) == 1:
        array_of_vectors = [array_of_vectors]
    
    # Iterate over each vector in the array
    for v2 in array_of_vectors:
        # Convert the current vector to a numpy array
        v2 = np.array(v2)
        # Compute the dot product of v1 and v2
        dot_product = np.dot(v1, v2)
        # Compute the norms of the vectors
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        # Compute the cosine similarity and append to the list
        similarity = dot_product / (norm_v1 * norm_v2)
        similarities.append(similarity)
    return np.array(similarities)

def euclidean_distance(v1, array_of_vectors):
    """
    Compute the Euclidean distance between a vector and an array of vectors.
    
    Parameters:
    v1 (array-like): The first vector.
    array_of_vectors (array-like): An array of vectors or a single vector.

    Returns:
    list: A list of Euclidean distances between v1 and each vector in array_of_vectors.
    """
    # Ensure that v1 is a numpy array
    v1 = np.array(v1)
    # Initialize a list to store distances
    distances = []
    
    # Check if array_of_vectors is a single vector
    if len(np.shape(array_of_vectors)) == 1:
        array_of_vectors = [array_of_vectors]
    
    # Iterate over each vector in the array
    for v2 in array_of_vectors:
        # Convert the current vector to a numpy array
        v2 = np.array(v2)
        # Check if the input arrays have the same shape
        if v1.shape != v2.shape:
            raise ValueError(f"Shapes don't match: v1 shape: {v1.shape}, v2 shape: {v2.shape}")
        # Calculate the Euclidean distance and append to the list
        dist = np.sqrt(np.sum((v1 - v2) ** 2))
        distances.append(dist)
    return distances


def pprint(*args, **kwargs):
    kwargs.setdefault('sort_dicts', False)
    original_pprint(*args, **kwargs)
    
def generate_with_single_input(prompt: str, 
                               role: str = 'user', 
                               top_p: float = 1, 
                               temperature: float = 0,
                               max_tokens: int = 500,
                               model: str ="gpt-4o",
                               openai_api_key: str = None,
                              **kwargs):
    
    # Use provided key or fall back to environment variable
    if openai_api_key is None:
        openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key is None:
        raise Exception("OpenAI API key not found. Provide `openai_api_key` or set `OPENAI_API_KEY` env variable.")

    # Configure client
    client = openai.OpenAI(api_key=openai_api_key)    

    payload = {
            "model": model,
            "messages": [{'role': role, 'content': prompt}],
            "top_p": top_p,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs

                  }
    try:
        response = client.chat.completions.create(**payload)
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"Failed to call OpenAI LLM. Error: {e}")
    

# Read the CSV without parsing dates

def read_dataframe(path):
    df = pd.read_csv(path, index_col=0)

    # Convert the DataFrame to dictionary after formatting
    df= df.to_dict(orient='records')
    return df


