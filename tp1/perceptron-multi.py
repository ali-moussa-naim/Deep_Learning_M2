import matplotlib.pyplot as plt
import numpy as np
import csv

models = []

def main():
    filename = "dataset-multi.csv"
    features = loadfeatures(filename)
    train(model, features, alpha)
    #readfeatures(features)
    displayDataset(features, model)

if __name__ == "__main__":
    main()
