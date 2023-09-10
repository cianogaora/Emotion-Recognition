import os
import shutil

def split_data():
    emotes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    i = 0
    train_labels = []
    if os.path.isdir('all_test'):
        shutil.rmtree('all_test')

    os.mkdir('all_test')
    for emote in emotes:
        path = f'archive/test/{emote}'
        num_ims = len(os.listdir(path))
        print(f'adding images from {path}')
        for im_name in os.listdir(path):
            shutil.copy(path + '/' + im_name, 'all_test/' + 'e' + str(i) + 'e' + im_name)

        for x in range(num_ims):
            train_labels.append(str(i))
        print(f'{emote}: {num_ims}')
        i += 1

    # return train_labels


if __name__ == '__main__':
    split_data()
