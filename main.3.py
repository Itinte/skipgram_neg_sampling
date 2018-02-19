import skipGram
import numpy as np

# path of the input file
#path = '/home/quentinbb/class/nlp/NLP_Project/Project_1/input-1000.txt'
path = 'C:/Users/Louis/Documents/AM2014-2015-2016/2017-2018/Essec-Centrale_Paris/NLP/NLP_Project/Project_1/input-1000.txt'
# path of the stop word file
#stopwords_path = '/home/quentinbb/class/nlp/NLP_Project/Project_1/stopwords.csv'
stopwords_path = 'C:/Users/Louis/Documents/AM2014-2015-2016/2017-2018/Essec-Centrale_Paris/NLP/NLP_Project/Project_1/stopwords.csv'

# load the stopwords file
stopwords = skipGram.loadStopwords(stopwords_path)

# load the text and get the sentences
sentences = skipGram.text2sentences(path, stopwords)

# initialize the skipgram
skipmodel = skipGram.mySkipGram(sentences)

# train it
# the first argument is the step size (or learning rate) in gradient descent
# the second argument is the number of epochs
skipmodel.train(0.4, 10)

# path of the folder where you want to save
# skipmodel.save('/home/quentinbb/class/nlp/model_saved')
skipmodel.save('C:/Users/Louis/Documents/AM2014-2015-2016/2017-2018/Essec-Centrale_Paris/NLP')


#skipmodel2 = skipmodel.load('/home/quentinbb/class/nlp/model_saved')
skipmodel2 = skipmodel.load('C:/Users/Louis/Documents/AM2014-2015-2016/2017-2018/Essec-Centrale_Paris/NLP')

print("-> Displaying some random similarity example:")
# do only the first 'limit'-2-uplets to check
limit = 100  # how many 2-uplet similarity you want to print
counter = 0

for i in range(limit):
    a = np.random.randint(len(skipmodel2.vocabulary_list_new))
    b = np.random.randint(len(skipmodel2.vocabulary_list_new))

    a_item = skipmodel2.vocabulary_list_new[a]
    b_item = skipmodel2.vocabulary_list_new[b]

    print(a_item, b_item, skipmodel2.similarity(a_item, b_item))

    counter += 1
    if counter > limit:
        break
