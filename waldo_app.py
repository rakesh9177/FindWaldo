import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import gradio as gr

def scale_image(image, alpha=1):
    # Get dimensions of the original image
    height, width = image.shape[:2]
    # Scale width and height using alpha parameter
    new_height = int(height * alpha)
    new_width = int(width * alpha)
    # Resize the image using cv2.resize()
    scaled_image = cv2.resize(image, (new_width, new_height))
    return scaled_image

def conv2d_crosscorrelation_coeff(image, kernel, strides = 1):
    x_img_shape, y_img_shape = image.shape
    x_kern_shape, y_kern_shape = kernel.shape
    xOutput = int(((x_img_shape - x_kern_shape ) / strides) + 1)
    yOutput = int(((y_img_shape - y_kern_shape ) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    mean_kernel = np.mean(kernel)
    std_kernel = np.std(kernel)
    
    for y in range(image.shape[1]):
        if y > image.shape[1] - y_kern_shape:
            break
        for x in range(image.shape[0]):
            if x > image.shape[0] - x_kern_shape:
                break
            img_slice = image[x:x+x_kern_shape, y:y+y_kern_shape]
            
            mean_img_slice = np.mean(img_slice)
            std_img_slice = np.std(img_slice)
            #coeff = np.sum((img_slice - mean_img_slice) * (kernel - mean_kernel)) / (std_img_slice * std_kernel)
            if std_img_slice > 1e-6 and std_kernel > 1e-6:
                coeff = np.sum((img_slice - mean_img_slice) * (kernel - mean_kernel)) / (std_img_slice * std_kernel)
            else:
                coeff = 0
            
            #output[x,y] = (kernel * img_slice).sum()
            output[x,y] = coeff
    return output

def find_waldo_demo_convolution(image, template, alpha=1):
    main_image = image.copy()
    main_image = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
    template_image = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template_image = scale_image(template_image, alpha)
    result = conv2d_crosscorrelation_coeff(main_image, template_image)
    max_loc = np.unravel_index(np.argmax(result), result.shape)
    waldo_location_fft = max_loc
    h, w = template_image.shape
    top_left = waldo_location_fft[::-1]
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 3)
    return image

def find_waldo_demo_fft(image, template, alpha=1):
    # Perform FFT on both the main image and the template
    main_image = image.copy()
    main_image = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
    template_image = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template_image = scale_image(template_image, alpha)
    image_fft = np.fft.fft2(main_image)
    template_fft = np.fft.fft2(template_image, s=image_fft.shape)

    corr_fft = np.real(np.fft.ifft2((image_fft * np.conj(template_fft))/ np.abs(image_fft*np.conj(template_fft))))
    # Find the location with the maximum correlation coefficient
    max_loc = np.unravel_index(np.argmax(corr_fft), corr_fft.shape)

    waldo_location_fft = max_loc

    h, w = template_image.shape
    top_left = waldo_location_fft[::-1]
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(image,top_left, bottom_right, (0, 255, 0), 3)
    return image

def selected_function(choice, image, template, alpha):
    if choice == "FFT":
        return find_waldo_demo_fft(image, template, alpha)
    elif choice == "Convolution":
        return find_waldo_demo_convolution(image, template, alpha)
    else:
        raise ValueError("Invalid choice")
    
dropdown = gr.Dropdown(choices=["FFT", "Convolution"], label="Select function")
alpha_slider = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, label="Scale Factor")

demo = gr.Interface(fn=selected_function, inputs=[dropdown, "image","image", alpha_slider], outputs="image")
demo.launch()

