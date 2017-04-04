# rnn_demo

Implementation of LSTM and GRU. 

You can building the project by Maven 3.0.

## Data description

-Dictionary: toy.memeformat.dict. [word_id, word]

-Sentences: toy.memeformat. [sentence_id, word_id, (token: no functional)]

## Dependencies

Guava (mentioned in pom.xml) cannot be accessed in China by GFW, you may need vpn to completely download it by Maven. Otherwise, you can manually add guava_18.0 which I have included into this project.
