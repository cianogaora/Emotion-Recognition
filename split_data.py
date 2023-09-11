import os
import shutil

def split_data(choice):
    if choice == 1:
        folder_name = 'all_train'
    else:
        folder_name = 'all_test'
    emotes = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    i = 0
    train_labels = []
    if os.path.isdir(folder_name):
        shutil.rmtree(folder_name)

    os.mkdir(folder_name)
    for emote in emotes:
        path = f'archive/{folder_name[4:]}/{emote}'
        num_ims = len(os.listdir(path))
        print(f'adding images from {path}')
        for im_name in os.listdir(path):
            shutil.copy(path + '/' + im_name, folder_name + '/' + 'e' + str(i) + 'e' + im_name)

        for x in range(num_ims):
            train_labels.append(str(i))
        print(f'{emote}: {num_ims}')
        i += 1

    # return train_labels


if __name__ == '__main__':
    choice = int(input("Enter data split choice: "))
    split_data(choice)
