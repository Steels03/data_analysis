from os.path import exists

data_file = "project/dataset/index.html"

if (not exists(data_file)):
    print("ERROR : " + data_file + " isn't present")
    exit(0)
else:
    print("dataset is present")