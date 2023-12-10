import cv2
import numpy as np
import enchant
from googletrans import Translator

emnist_labels = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122]


def letters_extract(image_file: str, out_size=28):
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # преобразование в чб
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY) # пороговая обработка
    img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1) # уменьшение белых областей на изображении

    # получение контуров и иерархий
    contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    output = img.copy()

    letters = []
    for idx, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour) # левый верхний угол прямоугольника, его ширина и высота
        if hierarchy[0][idx][3] == 0:
            cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1) # прямоугольник для каждого контура
            letter_crop = gray[y:y + h, x:x + w] # только контур в прямоугольнике
            # центр в середине изображения
            size_max = max(w, h)
            letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
            if w > h:
                y_pos = size_max // 2 - h // 2
                letter_square[y_pos:y_pos + h, 0:w] = letter_crop
            elif w < h:
                x_pos = size_max // 2 - w // 2
                letter_square[0:h, x_pos:x_pos + w] = letter_crop
            else:
                letter_square = letter_crop

            # приведение буквы к размеру 28x28 и добавление буквы с х-координатой
            letters.append((x, w, cv2.resize(letter_square, (out_size, out_size), interpolation=cv2.INTER_AREA)))

    # сортировка списка по х-координате
    letters.sort(key=lambda x: x[0], reverse=False)

    '''
    cv2.imshow("Input", img)
    cv2.imshow("Enlarged", img_erode)
    cv2.imshow("Output", output)
    cv2.waitKey(0)
    '''
    return letters


def emnist_predict_img(model, img):
    img_arr = np.expand_dims(img, axis=0) # список представляющий изображение, путем добавления одного измерения в начало (ось 0)
    img_arr = 1 - img_arr / 255.0 # инвертируются цвета изображения для нормализации значений пикселей в диапазоне от 0 до 1
    img_arr[0] = np.rot90(img_arr[0], 3) # поворот на 90 градусов влево, т.к изображения в датасете повернуты
    img_arr[0] = np.fliplr(img_arr[0]) # изображение в списке отражается горизонтально (зеркальное отражение)
    img_arr = img_arr.reshape((1, 28, 28, 1)).astype("float32")
    predict = model.run(None, {model.get_inputs()[0].name: img_arr})[0]
    result = np.argmax(predict, axis=1) # макс значение по оси
    return chr(emnist_labels[result[0]]) # возврат символа


def img_to_str(model, image_file: str):
    letters = letters_extract(image_file)
    s_out = ""
    for i in range(len(letters)):
        dn = letters[i + 1][0] - letters[i][0] - letters[i][1] if i < len(letters) - 1 else 0 # расстояние между текущей и след буквой
        s_out += emnist_predict_img(model, letters[i][2])
        if (dn > letters[i][1] / 5): # добавление пробелов
            s_out += ' '
    return s_out


def replace_unknown_words(s_out):
    dictionary = enchant.Dict("en_US")
    words = s_out.split()
    new_s_out = []
    for word in words:
        if dictionary.check(word):
            new_s_out.append(word)
        else:
            suggestions = dictionary.suggest(word)
            if suggestions:
                new_word = suggestions[0]
                new_s_out.append(new_word)
            else:
                new_s_out.append(word)

    updated_text = ' '.join(new_s_out)
    return updated_text


def translator(s_out):
    translator = Translator()
    text_Rus = translator.translate(s_out, dest='ru')
    return f"{text_Rus.text}"


def recognition(model, image_file):
    s_out = img_to_str(model, image_file)
    s_out = s_out.replace('1', 'l').replace('0', 'o').replace('3', 'a')
    s_out = s_out[0] + s_out[1:].lower()
    s_out = replace_unknown_words(s_out)
    return s_out



