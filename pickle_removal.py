import pickle
from matplotlib import pyplot as plt

infile = open('pickle.pkl', 'rb')
content = pickle.load(infile)
infile.close()
print("Finished depickling the pickle file.")

for key in content:
    print(key)

content['relevant_testing_results']