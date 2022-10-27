import cv2
import os
import shutil

def main():
    emotes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    i = 0
    train_labels = []
    if os.path.isdir('all_train'):
        shutil.rmtree('all_train')

    os.mkdir('all_train')
    for emote in emotes:
        path = f'archive/train/{emote}'
        num_ims = len(os.listdir(path))

        for im_name in os.listdir(path):
            shutil.copy(path + '/' + im_name, 'all_train/' + 'e' + str(i) + 'e' + im_name)

        for x in range(num_ims):
            train_labels.append(str(i))
        print(f'{emote}: {num_ims}')
        i += 1

    # return train_labels


if __name__ == '__main__':
    main()
