import numpy as np
import pickle
import nltk

def prepareWordTokenized(seedName):
    inputName = seedName + ".txt"
    
    doLowercase = True
    vocabSize = 10000
    unknown_token = "UNKNOWN_TOKEN"
    
    
    with open(inputName, "r") as fp:
        text = fp.read()
    
    tokensRaw = nltk.word_tokenize(text)
    if doLowercase:
        tokens = [w.lower() for w in tokensRaw]
    else:
        tokens = tokensRaw
    fdist = nltk.FreqDist(word for word in tokens)
    vocab = fdist.most_common(vocabSize - 1)
    
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    index_to_word = dict(enumerate(index_to_word))
    word_to_index = {v: k for k, v in index_to_word.items()}
    
    vocabSize = len(word_to_index)
    textLength = len(tokens)
    
    # Replace unknown words with unknown_tokens
    tokensReplaced = [w if w in word_to_index else unknown_token for w in tokens]
    
    hotText = np.zeros((textLength, vocabSize), dtype="float32")
    for index in range(textLength):
        hotText[index, word_to_index[tokensReplaced[index]]] = 1
    
    pickleFilename = seedName + ".pkl"
    with open(pickleFilename, "wb") as fp:
        pickle.dump((hotText, index_to_word, word_to_index), fp)
    # exampleLength = 24
    # nExamples = textLength - exampleLength
    
    # x = np.zeros((nExamples, exampleLength, vocabSize), dtype=int)
    # y = np.zeros((nExamples, exampleLength, vocabSize), dtype=int)
    # pointer = 0
    
    # for index in range(nExamples):
    #     x[index] = hotText[pointer: pointer + exampleLength]
    #     y[index] = hotText[pointer + 1: pointer + exampleLength + 1]
    #     pointer += 1
    
    # pickleFilename = seedName + "Matrix.pkl"
    # with open(pickleFilename, "wb") as fp:
    #     pickle.dump((x, y, index_to_word, word_to_index), fp)

def prepareCharacterTokenized(seedName):
    inputName = seedName + ".txt"
    with open(inputName, "r") as fp:
        text = fp.read()
    
    tokens = list(text)
    index_to_word = [chr(index) for index in range(128)]
    index_to_word = dict(enumerate(index_to_word))
    word_to_index = {v: k for k, v in index_to_word.items()}
    
    vocabSize = len(word_to_index)
    textLength = len(tokens)
    #------ Replace unknown chard with substitute (ASCII no 26)
    tokensReplaced = [w if w in word_to_index else index_to_word[26] for w in tokens]
    
    hotText = np.zeros((textLength, vocabSize), dtype="float32")
    for index in range(textLength):
        hotText[index, word_to_index[tokensReplaced[index]]] = 1
    
    pickleFilename = seedName + "Letters.pkl"
    with open(pickleFilename, "wb") as fp:
        pickle.dump((hotText, index_to_word, word_to_index), fp)

def f2(indexf2, index_to_word):
    print(index_to_word[np.argmax(hotText[indexf2])])
