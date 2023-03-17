import numpy as np
import matplotlib.pyplot as plt

#The following lines take in and display an Image of Einstein.
image_arr = np.loadtxt('e.txt')
print(image_arr.shape)

plt.imshow(image_arr, cmap='gray')
ax = plt.gca() # get current axis
ax.axes.xaxis.set_visible(False) # hide ticks and axis labels
ax.axes.yaxis.set_visible(False) # hide ticks and axis labels
plt.title('Image')
plt.colorbar()      # Show the colorbar scale by default

def gaussian(m, n, sigma):
    return np.exp(-1*(m*m+n*n)/(2*sigma*sigma))


def fft_of_gaussian(sigma, loc):
    plt.figure(figsize=(10,6))
    gaussian_arr = np.zeros(shape=(265,225))
    gaussian_sum = 0;
    for m in range(-132, 133, 1):
        for n in range(-112, 113, 1):
            gmn = gaussian(m, n, sigma)
            gaussian_arr[m,n] = gmn
            gaussian_sum += gmn
        
    gaussian_arr /= gaussian_sum

    #The following lines show the Gaussian convolution kernel and its Fourier transform.
    plt.subplot(2, 2, 1 + 2*loc)
    plt.imshow(gaussian_arr, cmap='gray')
    ax = plt.gca() # get current axis
    ax.axes.xaxis.set_visible(False) # hide ticks and axis labels
    ax.axes.yaxis.set_visible(False) # hide ticks and axis labels
    plt.title(f'Gaussian Convolution \nKernel with sigma = {sigma}')
    plt.colorbar()      # Show the colorbar scale by default

    conv_fft = np.fft.fft2(gaussian_arr)

    max_abs = 0
    for m in range(265):
        for n in range(225):
            if np.abs(conv_fft[m,n].imag) > max_abs:
                max_abs = np.abs(conv_fft[m,n].imag)

    conv_fft_real = conv_fft.real
    plt.subplot(2, 2, 2 + 2*loc)
    plt.imshow(conv_fft_real, cmap='gray')
    ax = plt.gca() # get current axis
    ax.axes.xaxis.set_visible(False) # hide ticks and axis labels
    ax.axes.yaxis.set_visible(False) # hide ticks and axis labels
    plt.title(f'Fourier Transform of \nConvolution Kernel \nwith sigma = {sigma}')
    plt.colorbar()      # Show the colorbar scale by default
    
    return gaussian_arr, conv_fft

conv_ker, conv_fft = fft_of_gaussian(3, 0)
fft_of_gaussian(10, 1)

#The following lines produce a blurred image of Einstein.
image_fft = np.fft.fft2(image_arr)

fft_prod = image_fft * conv_fft
invfft = np.fft.ifft2(fft_prod)
invfft_real = invfft.real



plt.imshow(invfft_real, cmap='gray')
ax = plt.gca() # get current axis
ax.axes.xaxis.set_visible(False) # hide ticks and axis labels
ax.axes.yaxis.set_visible(False) # hide ticks and axis labels
plt.title('Convolved Image')
plt.colorbar()      # Show the colorbar scale by default

#The following lines produce a grid of random noise and displays it.

random_grid = np.zeros(shape=(265, 225))
for m in range(265):
    for n in range(225):
        random_grid[m,n] = np.random.normal(0, 0.01)
        
plt.imshow(random_grid, cmap='gray')
ax = plt.gca() # get current axis
ax.axes.xaxis.set_visible(False) # hide ticks and axis labels
ax.axes.yaxis.set_visible(False) # hide ticks and axis labels
plt.title('Random Noise')
plt.colorbar()      # Show the colorbar scale by default

#The following lines show three deconvolved images of Einstein for different thresholds,
#to avoid the inaccuracy caused by division by zero.
plt.figure(figsize=(10,6))

fft_of_blurred_2 = np.fft.fft2(invfft)
quotient_2 = np.zeros(shape = (265, 225), dtype = 'complex_')
for m in range(265):
    for n in range(225):
        if conv_fft[m,n] > 0.1:
            quotient_2[m,n] = fft_of_blurred_2[m,n] / conv_fft[m,n]
        else:
            quotient_2[m,n] = fft_of_blurred_2[m,n]
deconv_2 = np.fft.ifft2(quotient_2)
deconv_2_real = deconv_2.real

plt.subplot(1, 3, 1)
plt.imshow(deconv_2_real, cmap='gray')
ax = plt.gca() # get current axis
ax.axes.xaxis.set_visible(False) # hide ticks and axis labels
ax.axes.yaxis.set_visible(False) # hide ticks and axis labels
plt.title('Deconvolved Image 2')
plt.colorbar()      # Show the colorbar scale by default

fft_of_blurred_3 = np.fft.fft2(invfft)
quotient_3 = np.zeros(shape = (265, 225), dtype = 'complex_')
for m in range(265):
    for n in range(225):
        if conv_fft[m,n] > 0.001:
            quotient_3[m,n] = fft_of_blurred_3[m,n] / conv_fft[m,n]
        else:
            quotient_3[m,n] = fft_of_blurred_3[m,n]
deconv_3 = np.fft.ifft2(quotient_3)
deconv_3_real = deconv_3.real

plt.subplot(1, 3, 2)
plt.imshow(deconv_3_real, cmap='gray')
ax = plt.gca() # get current axis
ax.axes.xaxis.set_visible(False) # hide ticks and axis labels
ax.axes.yaxis.set_visible(False) # hide ticks and axis labels
plt.title('Deconvolved Image 3')
plt.colorbar()      # Show the colorbar scale by default

fft_of_blurred_4 = np.fft.fft2(invfft)
quotient_4 = np.zeros(shape = (265, 225), dtype = 'complex_')
for m in range(265):
    for n in range(225):
        if conv_fft[m,n] > 1.e-4:
            quotient_4[m,n] = fft_of_blurred_4[m,n] / conv_fft[m,n]
        else:
            quotient_4[m,n] = fft_of_blurred_4[m,n]
deconv_4 = np.fft.ifft2(quotient_4)
deconv_4_real = deconv_4.real

plt.subplot(1, 3, 3)
plt.imshow(deconv_4_real, cmap='gray')
ax = plt.gca() # get current axis
ax.axes.xaxis.set_visible(False) # hide ticks and axis labels
ax.axes.yaxis.set_visible(False) # hide ticks and axis labels
plt.title('Deconvolved Image 4')
plt.colorbar()      # Show the colorbar scale by default

plt.imshow(invfft_real, cmap='gray')
ax = plt.gca() # get current axis
ax.axes.xaxis.set_visible(False) # hide ticks and axis labels
ax.axes.yaxis.set_visible(False) # hide ticks and axis labels
plt.title('Convolved Image')
plt.colorbar()      # Show the colorbar scale by default


#The following lines take in the image of Marilyn Monroe, 
# convolve it, and display the convolved image.
m = np.loadtxt('m.txt')

m_fft = np.fft.fft2(m)

fft_prod_m = m_fft * conv_fft
invfft_m = np.fft.ifft2(fft_prod_m)
invfft_real_m = invfft_m.real



plt.imshow(invfft_real_m, cmap='gray')
ax = plt.gca() # get current axis
ax.axes.xaxis.set_visible(False) # hide ticks and axis labels
ax.axes.yaxis.set_visible(False) # hide ticks and axis labels
plt.title('Convolved Image')
plt.colorbar()      # Show the colorbar scale by default

#The following lines show a 'high-pass' image of Einstein.

high_pass = image_arr - invfft
high_pass_real = high_pass.real

plt.imshow(high_pass_real, cmap='gray')
ax = plt.gca() # get current axis
ax.axes.xaxis.set_visible(False) # hide ticks and axis labels
ax.axes.yaxis.set_visible(False) # hide ticks and axis labels
plt.title('High Pass Image')
plt.colorbar()      # Show the colorbar scale by default

#The following lines produce the sum of the high-pass Einstein
#  and convolved Monroe, and display the composite image.

image_sum = high_pass + invfft_m
image_sum_real = image_sum.real

plt.imshow(image_sum_real, cmap='gray')
ax = plt.gca() # get current axis
ax.axes.xaxis.set_visible(False) # hide ticks and axis labels
ax.axes.yaxis.set_visible(False) # hide ticks and axis labels
plt.title('Image Sum')
plt.colorbar()      # Show the colorbar scale by default