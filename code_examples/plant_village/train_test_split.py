import os
from shutil import copyfile


def split(source):
    for file in os.listdir(source):
        os.mkdir("C:\\Users\\berkt\\PycharmProjects\\ANN\\test\\" + file)
        os.mkdir("C:\\Users\\berkt\\PycharmProjects\\ANN\\train\\" + file)
        for index,picture in enumerate(os.listdir(source + "\\" + file)):
            if index <= len(os.listdir(source + "\\" + file))/4:
                copyfile(source + "\\" + file + "\\" + picture, "C:\\Users\\berkt\\PycharmProjects\\ANN\\test\\" + file + "\\" + picture)
            else:
                copyfile(source + "\\" + file + "\\" + picture, "C:\\Users\\berkt\\PycharmProjects\\ANN\\train\\" + file + "\\" + picture)



if __name__ == '__main__':
    split('C:\\Users\\berkt\\PycharmProjects\\ANN\\Tomato')


