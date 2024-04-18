import os
import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import turtle

from tensorflow import keras
from main import ExportModel
from ocr_train import ExportOCRModel
from PIL import Image, ImageChops


t=turtle


def main():
    # model = keras.models.load_model('handwriting_model') # up to 75%
    # model = keras.models.load_model('handwriting_model_long') # up to 80%
    model = keras.models.load_model('handwriting_model_omega') # up to 90%
    # model = keras.models.load_model('handwriting_model_supalong') # broken model
    handwriting_model = ExportModel(model)

    ocr_model = keras.models.load_model('OCR_model')
    ocr_export_model = ExportOCRModel(ocr_model)

    t.title("PyBoard")
    t.setup(1000, 1000)
    t.screensize(1000, 1000)
    t.shape("circle")
    t.shapesize(1.5)
    t.pu()
    t.color("black") # dot
    t.bgcolor("white")  # background
    t.pencolor("black") # line
    t.pensize(30)
    t.speed(0)

    # move pen to mouse
    def skc(x,y):
        t.pu()
        t.goto(x,y)
        def sketch(x,y):
            t.pd()
            t.goto(x,y)
        t.ondrag(sketch)
    t.onscreenclick(skc)

    # eraser
    def erase():
        t.speed(0)
        t.pencolor("white")
        t.pensize(40)
        t.shape("square")
        t.shapesize(2)

    # set colour
    def colr():
        t.speed(0)
        col=t.textinput("COLOR","SET COLOR")
        t.pencolor(col)
        t.color(col)
        if col=="": t.pencolor("black")
        t.listen()

    # clear canvas
    def clear():
        t.clear()

    # pen
    def backtopen():
        t.speed(0)
        t.color("black")
        t.pensize(30)
        t.shape("circle")
        t.shapesize(1.5)

    # undo last stroke
    def undo():
        t.undo()

    # save image
    def save_own():
        # get image and convert to png
        DATA_DIR = 'img.png'

        canvas = t.getscreen().getcanvas()
        canvas.postscript(file="foo.ps")
        psimage = Image.open("foo.ps")
        psimage = psimage.resize((100, 100), resample=Image.NEAREST)
        psimage.save(DATA_DIR)
        psimage.close()
        os.remove("foo.ps")

        # crop image
        pngimage = Image.open(DATA_DIR)
        ori = pngimage  # set image to original
        pngimage = trim(pngimage)   # trim away padding white pixels
        tw, th = pngimage.size
        # image wanted 64x64
        # if th=64, then 192-64 = 128 (no cropping occurs)
        # if th=65, then 192-65 = 127 (reduce original by 1)
        # if th=63, then 192-63 = 129 (increase original by 1)
        # resize with height only, awoid stretching
        pngimage = ori.resize((192-th, 192-th), resample=Image.NEAREST) 
        pngimage = trim(pngimage) # trim to get 64x64 image

        # create new image with dimension [128, 128, 1]
        blank_img = Image.new(mode="RGB", size=(128, 128), color=(255, 255, 255))

        # place image in center
        iw, ih = pngimage.size
        blank_img.paste(pngimage, (round(64-iw/2), round(64-ih/2)))

        # save file and close
        blank_img.save(DATA_DIR)
        pngimage.close()
        blank_img.close()

        data_dir = pathlib.Path(DATA_DIR)
        x = tf.io.read_file(str(data_dir))
        x = tf.image.decode_image(x)
        x = tf.image.rgb_to_grayscale(x)
        image = x
        x = x[tf.newaxis, ...]
        print(x.shape)

        result = handwriting_model(x)
        print(result)
        index = tf.argmax(result[0]).numpy()
        print("Prediction: ", handwriting_model.classes[index], " || ", handwriting_model.values[index])

        plt.figure(figsize=(20, 8))
        plt.subplot(1, 2, 1)
        plt.bar(handwriting_model.values, tf.nn.softmax(result[0]))
        plt.title(handwriting_model.values[index])

        plt.subplot(1, 2, 2)
        np_data = np.asarray(image)
        plt.imshow(np_data)
        plt.title(handwriting_model.values[index])

        plt.show()

    def save_ocr():
        # get image and convert to png
        DATA_DIR = 'img_ocr.png'

        canvas = t.getscreen().getcanvas()
        canvas.postscript(file="foo.ps")
        psimage = Image.open("foo.ps")
        psimage.save(DATA_DIR)
        psimage.close()
        os.remove("foo.ps")

        # crop image
        pngimage = Image.open(DATA_DIR)
        pngimage = trim(pngimage)   # trim away padding white pixels
        pngimage.save(DATA_DIR)
        pngimage.close()

        result = ocr_export_model.pred(DATA_DIR)
        print(result)
        
    # start
    t.onkey(colr,"C")
    t.onkeypress(undo,"u")
    t.onkey(backtopen,"p")
    t.onkey(clear,"c")
    t.onkey(erase,"e")
    t.onkey(save_own, "S")
    t.onkey(save_ocr, "O")
    t.listen()
    t.mainloop()


def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, 0)
    #Bounding box given as a 4-tuple defining the left, upper, right, and lower pixel coordinates.
    #If the image is completely empty, this method returns None.
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)


def test_model():
    # DATA_DIR = "D:\\training_data\\handwriting\\data\\by_class\\4a\\4a_hsf_0_00000.png"
    DATA_DIR = "C:\\Users\\User\\Desktop\\Python\\mnlt\\handwriting\\img.png"
    model = keras.models.load_model('handwriting_model')

    data_dir = pathlib.Path(DATA_DIR)
    x = tf.io.read_file(str(data_dir))
    x = tf.image.decode_image(x)
    x = tf.image.rgb_to_grayscale(x)
    x = x[tf.newaxis, ...]
    print(x.shape)

    # from image
    exported_model = ExportModel(model)
    pred = exported_model(x)
    index = tf.argmax(pred[0]).numpy()
    print("Prediction: ", exported_model.classes[index], " || ", exported_model.values[index])

    # from path
    from_str = exported_model(DATA_DIR)
    index = tf.argmax(from_str[0]).numpy()
    print("Prediction: ", exported_model.classes[index], " || ", exported_model.values[index])

    plt.figure(figsize=(16, 8))
    x_label = exported_model.values
    plt.bar(x_label, tf.nn.softmax(pred[0]))
    plt.title(x_label[index])
    plt.show()


if __name__ == '__main__':
    # test_model()
    main()
