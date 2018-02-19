# NLP_Project
Project_1 : Skip Gram computation

The project aims at modeling Skip Gram  algorithm with negative sampling method

## Run

In `main.3.py` you have to :
- Set the input text path, here our input text is `input-1000.txt`
- Set the input stop word file path, here our stop word file is `stopwords.csv`
- Set the path where you want to save your files

## Tune parameters

In `main.3.py` you can tune stepSize (learning rate in the gradient descent) and number of epoch in `skipmodel.train(stepSize,epoch)`.
In `skipGram.py` you can tune Context words window `winSize`, the number of Embedded dimension `nEmbed`, the number of negative sampling for one true example `negativeRate`and the number of time a word is at least counted `minCount`.

## Save

We save the Weights associated to contexts and words and the vocabulary list.

## Print

We print similiraty of 100 couples of words.

