import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
#set up the image path
image_path = './tester3.jpeg'


def gaussian_pyramid_pic(pic):
    # read image from path
    img = cv2.imread(pic)
    # create a copy of image
    lower = img.copy()

    # Create a Gaussian Pyramid
    gaussian_pyr = [lower]
    for i in range(2):
        lower = cv2.pyrDown(lower)
        gaussian_pyr.append(lower)

    # Resize the feature map to match the image input.Unwanted high-frequencies pixels are discarded
    ans = cv2.pyrUp(gaussian_pyr[-1])
    ans = cv2.pyrUp(ans)

    # overwrite the input with feature maps

    cv2.imwrite(pic, ans)


def colour_mapping():
    # read image from path

    image = cv2.imread(image_path)
    # convert image into HSV colour model

    HSV_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Hue (0 - 10) RED color mask

    lower_mask = cv2.inRange(HSV_image, (0, 100, 90), (19, 255, 255))
    # Hue (160 - 180) RED color mask

    upper_mask = cv2.inRange(HSV_image, (150, 100, 90), (179, 255, 255))
    # combine lower mask and upper mask
    full_mask = lower_mask + upper_mask

    # colour mask for green,yellow and blue

    mgreen = cv2.inRange(HSV_image, (40, 0, 90), (79, 255, 255))
    myellow = cv2.inRange(HSV_image, (20, 100, 90), (39, 255, 255))
    mblue = cv2.inRange(HSV_image, (80, 100, 90), (149, 255, 255))
    # store initial coloured segmented image according to colour

    cv2.imwrite('./colour/RED.png', full_mask)
    cv2.imwrite('./colour/BLUE.png', mblue)
    cv2.imwrite('./colour/YELLOW.png', myellow)
    cv2.imwrite('./colour/GREEN.png', mgreen)
    # merge all segmented image
    full_mask = full_mask + mgreen + myellow + mblue

    # store image for two copy, one use for gabor filter , one show pre-gabor filter result

    cv2.imwrite('./colour/colour map.png', full_mask)
    cv2.imwrite('./colour/colour map_pregabor.png', full_mask)

    gaussian_pyramid_pic('./colour/colour map.png')


def gray_scaling():
    # read image from path
    BGR_image = cv2.imread(image_path)
    # convert image into grayscale
    gray = cv2.cvtColor(BGR_image, cv2.COLOR_BGR2GRAY)
    # store image for two copy, one use for gabor filter , one show pre-gabor filter result
    cv2.imwrite('./intensity/grey map.png', gray)
    cv2.imwrite('./intensity/grey map_pregabor.png', gray)

    gaussian_pyramid_pic('./intensity/grey map.png')


def gabor_filters():
    alpha_value = 0.5  # local variable use to blend images

    ksize = 25  # Size of the filter returned
    sigma = 8  # Standard deviation of the gaussian envelope.

    lamda = 7  # wavelength of the sinusoidal factor.
    gamma = 0.3  # Spatial aspect ratio.
    phi = 0  # Phase offset.
    # set up different angle parameters
    theta = 0
    theta2 = 45
    theta3 = 90
    theta4 = 135
    # read image from path
    img_ori = cv2.imread(image_path)
    # convert image into grayscale

    img = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)

    # set up the kernel
    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)
    # filter the image using kernel
    f_img = cv2.filter2D(img, 16, kernel)
    #save the kernel for merging use
    name = './orientation/kernel1.png'
    cv2.imwrite(name, f_img)

    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta2, lamda, gamma, phi, ktype=cv2.CV_32F)
    f_img = cv2.filter2D(img, 16, kernel)
    name2 = './orientation/kernel2.png'
    cv2.imwrite(name2, f_img)
    #open kernel images
    image_object_1 = Image.open('./orientation/kernel1.png')
    image_object_2 = Image.open('./orientation/kernel2.png')
    #blend kernel image together
    kernel_0_45 = Image.blend(image_object_1, image_object_2, alpha_value)

    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta3, lamda, gamma, phi, ktype=cv2.CV_32F)
    f_img = cv2.filter2D(img, 16, kernel)
    cv2.imwrite(name, f_img)

    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta4, lamda, gamma, phi, ktype=cv2.CV_32F)
    f_img = cv2.filter2D(img, 16, kernel)
    cv2.imwrite(name2, f_img)

    image_object_1 = Image.open('./orientation/kernel1.png')
    image_object_2 = Image.open('./orientation/kernel2.png')

    kernel_90_135 = Image.blend(image_object_1, image_object_2, alpha_value)
    #blend all kernel images together
    kernel_all4 = Image.blend(kernel_0_45, kernel_90_135, alpha_value)

    i = kernel_all4.save("./orientation/orientation map.png")
    # store image for two copy, one use for gabor filter , one show pre-gabor filter result
    i = kernel_all4.save("./orientation/orientation map_pregabor.png")

    gaussian_pyramid_pic('./orientation/orientation map.png')


def image_mean():
    # Since all feature maps are same size, get dimensions of first image
    w, h = Image.open('./intensity/grey map.png').size
    a = Image.open('./intensity/grey map.png')
    b = Image.open('./colour/colour map.png')
    c = Image.open('./orientation/orientation map.png')
    # Create a numpy array to store values
    arr = np.zeros((h, w, 3), float)

    # Add on value into array to get the total
    imarr = np.array(a, dtype=float)
    arr = arr + imarr
    imarr = np.array(b, dtype=float)
    arr = arr + imarr
    imarr = np.array(c, dtype=float)
    arr = arr + imarr
    # divide total by 3 to get the mean
    arr = arr / 3
    # Round values in array and cast as 8-bit integer
    arr = np.array(np.round(arr), dtype=np.uint8)

    # View the image and save it
    out = Image.fromarray(arr, mode="RGB")
    out.save("Saliency map.png")
    out.show()


if __name__ == "__main__":
    gabor_filters() #orientation
    colour_mapping() #colour
    gray_scaling() #intensity
    image_mean() #calculate the mean and obtain saliency map

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    plt.show()
