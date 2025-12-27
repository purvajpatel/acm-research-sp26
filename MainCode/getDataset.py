import pandas as pd

# these are all the teacher models and all the 3 splits we'll have, now we can loop through every combination of these in a double for loop
models = ["cola", "mnli", "mrpc", "qnli", "qqp", "sst2"]
dataTypes = ["train", "validation", "test"]
samplesPerType = [32, 8, 20]

# this will be the dictionary holding the data per model and per split.
dfCollection = {}

# loop through each model first
for model in models:
    dfCollection[model] = {}
    # then loop through each split type
    for i in range(len(dataTypes)):
        usedDataType = dataTypes[i]
        # weirldly this one has both mismatched and matched subsets for the validation and test set, so we make sure to just change the
        # name to pass into the url to be correct.
        if(model == "mnli"):
            if(dataTypes[i] == "validation"):
                usedDataType = "validation_matched"
            elif(dataTypes[i] == "test"):
                usedDataType = "test_matched"
        # get the actual data
        theDataset = pd.read_parquet(f"hf://datasets/nyu-mll/glue/{model}/{usedDataType}-00000-of-00001.parquet")

        # drop the columns that's aren't needed; since we're mimicking the unsupervised learning, we drop the labels too
        theDataset.drop(columns=["idx", "label"], inplace = True)
        
        # randomly sample a small amount per split and model
        theDataset = theDataset.sample(n=samplesPerType[i], random_state = 42)

        # save it to the dictionary
        dfCollection[model][dataTypes[i]] = theDataset
        print(f"The one with model {model} and the one with data type {dataTypes[i]} is done!")

# helper function to get the dictionary
def getDataset():
    return dfCollection