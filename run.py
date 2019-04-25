from PIL import Image
import argparse
import matplotlib.pyplot as plt
from detector import scan, nms
from classifier import predict, init_clf


def plot(image, classified_boxes, window_size):
    """
    Plots the bounding boxes with class label and prediction score on top of the original image.
    :param image: input image as PIL image object 
    :param classified_boxes: list of tuples (x_min <int>, y_min <int>, class prediction <string>, prediction score <float>)
    """
    fig1 = plt.figure(dpi=400)
    ax1 = fig1.add_subplot(1,1,1) 
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.axis('off')
    for box in classified_boxes:
        x_min, y_min, x_max, y_max = box[0]-.5, box[1]-.5, box[0]+window_size[0]-.5, box[1]+window_size[1]-.5
        prediction, predict_score = box[2], box[3]
        ax1.text(x_min, y_min-3, "%s %d%%" % (prediction, predict_score*100), color="red", fontsize=3)
        x = [x_max, x_max, x_min, x_min, x_max]
        y = [y_max, y_min, y_min, y_max, y_max]
        line, = ax1.plot(x,y,color="red")
        line.set_linewidth(.5)
    fig1.savefig("classification.png")
    plt.show()
    return
 

def main():
    # Initialization of classifier
    clf = init_clf()
    # Load the image
    parser = argparse.ArgumentParser()                                               
    parser.add_argument("--input", "-i", type=str, required=False)
    link = parser.parse_args().input
    if link:
        image = Image.open(link)    
    else:
        image = Image.open("./dataset/detection-images/detection-2.jpg")
    # Hyperparameters
    window_size         = (20,20)
    stride              = (1,1)
    detect_score_threshold     = .85 # Minimum detection score to be considered a hit.
    iou_threshold       = .1 # Too high --> false positives occur.
    max_boxes           = 100
    # Ru(i)n everything
    boxes               = scan(image, window_size, stride, detect_score_threshold)
    boxes               = nms(boxes, window_size, max_boxes, iou_threshold)
    classified_boxes    = predict(image, boxes, window_size, clf)
    print("Detecting and classifying input image")
    plot(image, classified_boxes, window_size)


if __name__== "__main__":
    main()