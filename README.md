<html>
<body>
  <h3>Super-resolution using Real-ESRGAN and ESRGAN in Medical Images</h3>
<p align="justify">I have used Real-ESRGAN and ESRGAN models for enhancing the resolution of brain and cardiac magnetic resonance images. The work here is based on Real-ESRGAN <a href="https://github.com/xinntao/Real-ESRGAN">(GitHub Link)</a> and ESRGAN <a href="https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/esrgan">(GitHub Link)</a>. I have modified the codes from these repositories to train the models for medical images. For Real-ESRGAN and ESRGAN, I have changed the input and output channel to one. In the original work, they were proposed to enhance the reolution of real-world images with three channels. In this work, the generation of low-resolution images is also different. The changes are reflected in the given codes. In the Real-ESRGAN code repository folder, replace the files realesrgan_dataset.py, realesrgan_model.py, train_realesrgan_x4plus.yml and utils.py. To train the models, I have used four datasets, which are BraTS 2018, ADNI, ACDC and Sunnybrook Cardiac Dataset.</p>
<h4>Low-resolution Image Generation</h4>
  <p align="justify">
The datasets do not contain low-resolution (LR) images. So, I had to convert the high-resolution (HR) images into low-resolution ones for training the models. First, I performed fast Fourier transform (FFT) to convert one HR image from spatial domain to frequency domain, where I made the
inner portion of the 2D k−space zero. After that, I performed inverse fast Fourier transform  IFFT and finally
linear interpolation to convert one 256 × 256 HR image to 64 × 64 LR image.
  </p>
  <p align="justify">
    Some examples from BraTS dataset are shown in the following figure. The first column represents the HR images,
    the second column is of LR images. Then from the 3rd column, we have images from Real-
    ESRGAN, ESRGAN, bicubic and bilinear interpolation.
  </p>
  <!--<img src="https://drive.google.com/file/d/11iPSQWY2VJW8CDqxLWtWTKE95hxmli7K/view?usp=sharing" alt="Comparison among Real-ESRGAN, ESRGAN and interpolation methods" title="Comparison among Real-ESRGAN, ESRGAN and interpolation methods.">-->
  <a href="https://drive.google.com/uc?export=view&id=11iPSQWY2VJW8CDqxLWtWTKE95hxmli7K"><img src="https://drive.google.com/uc?export=view&id=11iPSQWY2VJW8CDqxLWtWTKE95hxmli7K" style="width: 650px; max-width: 100%; height: auto" title="Comparison among Real-ESRGAN, ESRGAN and interpolation methods" />
  

  
</body>
</html>
