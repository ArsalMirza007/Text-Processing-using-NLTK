# Import the transform_chunk function from your custom module (transforms)
from transforms import transform_chunk

# Define a chunked sentence as a list of tagged words
chunked_sentence = [('the', 'DT'), ('book', 'NN'), ('of', 'IN'), ('recipes', 'NNS'), ('is', 'VBZ'), ('delicious', 'JJ')]

# Call transform_chunk with trace=1 to show intermediate steps
result = transform_chunk(chunked_sentence, trace=1)

# The function will print the intermediate steps and return the final transformed result
print(result)
