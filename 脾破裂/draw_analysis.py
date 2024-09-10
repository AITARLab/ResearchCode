from PIL import Image, ImageColor, ImageDraw
import numpy as np
import matplotlib.pyplot as plt


def draw_mask(image, mask, color, alpha):
    """
    cover the image with certain color and alpha mask
    :param image: input target image, numpy or Image
    :param mask: 01 matrix of the same size as the image
    :param color: (R,G,B) or python color string 'black', 'white' and so on
    :param alpha: the transparency of the mask [0,1] float
    :return: the image with the mask (numpy or Image)
    """
    # convert the image to numpy array if it is an Image object
    if isinstance(image, Image.Image):
        image = np.array(image)
        isImage = 1
    else:
        isImage = 0

    # convert the color to RGB tuple if it is a string
    if isinstance(color, str):
        color = ImageColor.getrgb(color)

    # create a new array with the same shape as the image and fill it with the color
    color_array = np.zeros_like(image)
    color_array[:] = color

    # create a new array with the same shape as the image and fill it with the alpha value
    alpha_array = np.zeros_like(image)
    alpha_array[:] = alpha

    anti_mask = image - mask * image

    # multiply the mask with the alpha array to get the final alpha values for each pixel
    final_mask = mask * color_array + anti_mask

    # blend the image and the color array using the final alpha values
    image_with_mask = Image.blend(Image.fromarray(np.uint8(image)), Image.fromarray(np.uint8(final_mask)), alpha)

    # convert the image to the original style
    if isImage == 0:
        image_with_mask = np.array(image_with_mask)

    return image_with_mask


def draw_rectangle(image,rectangle,color,width):
    """
    draw the rectangle on the image with certain color and width
    :param image: input target image, Image
    :param rectangle: tuple(x_min,y_min,x_max,y_max)
    :param color: (R,G,B) or python color string 'black', 'white' and so on
    :param width:
    :return: the image with the rectangle (Image)
    """

    # convert the color to RGB tuple if it is a string
    if isinstance(color, str):
        color = ImageColor.getrgb(color)

    # draw rectangle
    draw = ImageDraw.Draw(image)
    draw.rectangle(rectangle, outline=color, width=width)
    return image


def image_bitwise_mask(image, mask):
    """
     Keep the masked area in the image, leaving the remaining areas black
    :param image: input target image, numpy or Image        # img = [128,256,3]
    :param mask: 01 matrix of the same size as the image    # mask = [128,256,1]
    :return:
    """
    # convert the image to numpy array if it is an Image object
    if isinstance(image, Image.Image):
        image = np.array(image)
        isImage = 1
    else:
        isImage = 0

    # create a new array with the same shape
    black_arry = np.zeros(image.shape)
    mask = np.repeat(mask, 3, axis=2)
    result = np.where(mask > 0, image, black_arry)

    # convert the image to the original style
    if isImage == 1:
        result = Image.fromarray(np.uint8(result))

    return result


