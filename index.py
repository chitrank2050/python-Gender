import nltk
import random
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import names
def genFeatures(word):
    return {"suffix 1":word[-1],"suffix 2":word[-2], "prefix 1":word[0],"prefix 2":word[1],"length":len(word)}
completeSet=([(name,'male') for name in names.words('male.txt')]+[(name,'female') for name in names.words('female.txt')])
random.shuffle(completeSet)
featureSets=[(genFeatures(n),gender) for (n,gender) in completeSet]
train_names,devtest_names,test_names=completeSet[1500:],completeSet[500:1500],completeSet[:500]
train_set=[(genFeatures(n),gender) for (n,gender) in train_names]
devtest_set=[(genFeatures(n),gender) for (n,gender) in devtest_names]
test_set=[(genFeatures(n),gender) for (n,gender) in test_names]
try:

    classifier = NaiveBayesClassifier.train(train_set)
except Exception as e:
    print str(e)

value=raw_input("Enter the name:")
guess=classifier.classify(genFeatures(value))
print guess
print nltk.classify.accuracy(classifier,devtest_set)

