""""Paint program by Dave Michell.

Subject: tkinter "paint" example
From: Dave Mitchell <davem@magnet.com>
To: python-list@cwi.nl
Date: Fri, 23 Jan 1998 12:18:05 -0500 (EST)

  Not too long ago (last week maybe?) someone posted a request
for an example of a paint program using Tkinter. Try as I might
I can't seem to find it in the archive, so i'll just post mine
here and hope that the person who requested it sees this!

  All this does is put up a canvas and draw a smooth black line
whenever you have the mouse button down, but hopefully it will
be enough to start with.. It would be easy enough to add some
options like other shapes or colors...

                                                yours,
                                                dave mitchell
                                                davem@magnet.com
"""

from PIL import Image as PILImage
from tkinter import *
# from PIL import ImageGrab
import cv2
import numpy as np
import io
from predict import MnistPredictor


b1 = "up"
xold, yold = None, None

# WEIGHTS_PATH = 'logistic_regression_weights.h5'
WEIGHTS_PATH = 'checkpoints/cnn_weights_10-0.03.h5'
mnist_predictor = MnistPredictor(WEIGHTS_PATH)

def main():
    root = Tk()
    root.title('Classify Digit')
    drawing_area = Canvas(root, width=200, height=200)
    drawing_area.pack()
    drawing_area.bind("<Motion>", motion)
    drawing_area.bind("<ButtonPress-1>", b1down)
    drawing_area.bind("<ButtonRelease-1>", b1up)
    classification_label = Label(root, text='Prediction: ')
    gobutton = Button(root, text='Predict', command=classify_canvas(drawing_area, root, classification_label))
    wipebutton = Button(root, text='Wipe Screen', command=wipe_canvas(drawing_area, classification_label))
    gobutton.pack()
    wipebutton.pack()
    classification_label.pack()
    root.mainloop()

def b1down(event):
    global b1
    b1 = "down"           # you only want to draw when the button is down
                          # because "Motion" events happen -all the time-

def classify_canvas(canvas, root, output_label):
    def classify_canvas_function():
        ps = canvas.postscript(colormode='color')
        im = PILImage.open(io.BytesIO(ps.encode('utf-8')))
        # im.save("thething.png")
        im = np.array(im) # convert to numpy array
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) # flatten RGB to greyscale
        im = 255-im # invert colours
        predicted_category = predict(im)
        output_label.config(text='Prediction: {}'.format(predicted_category))
    return classify_canvas_function

def wipe_canvas(canvas, output_label):
    def wipe_canvas_function():
        canvas.delete('all')
        output_label.config(text='Prediction: ')
    return wipe_canvas_function

def predict(im):
    # return 3
    return mnist_predictor.predict_image(im)

def b1up(event):
    global b1, xold, yold
    b1 = "up"
    xold = None           # reset the line when you let go of the button
    yold = None

def motion(event):
    if b1 == "down":
        global xold, yold
        if xold is not None and yold is not None:
            event.widget.create_line(xold,yold,event.x,event.y,smooth=TRUE, width=10)
                          # here's where you draw it. smooth. neat.
        xold = event.x
        yold = event.y

if __name__ == "__main__":
    main()