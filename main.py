import MovieLens_Data as mld
from load_split_data import *


def main():
    train_set, test_set, train_dl, test_dl = split_and_load_data()

    i = 0
    for data in train_dl:
        if i ==2:
            break
        print(data[:])
        i+=1

if __name__ == '__main__':
    main()
