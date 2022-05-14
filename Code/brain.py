import numpy as np
import nibabel as nib
from tqdm import tqdm 
from nibabel.processing import resample_to_output,resample_from_to
from skimage.morphology import remove_small_holes, binary_dilation, binary_erosion, ball
from skimage.measure import label, regionprops
from scipy.ndimage import zoom
from tensorflow.python.keras.models import load_model
import cv2
import grand


lung_model=load_model('Models/lung_model.h5',compile=False)
brain_model=load_model('Models/brain_bce.h5',compile=False)
retina_model=load_model('Models/retina.h5',compile=False)
liver_model=load_model('Models/liver.h5',compile=False)

def predict(img,clas,himg):
    y=1
    if(clas=="LUNG"):
        himg=grand.hmap(lung_model,himg)
        prediction = lung_model.predict(img)
        prediction_image = prediction.reshape((256,256))
        
    elif(clas=="BRAIN"):
        himg=grand.hmap(brain_model,himg)
        prediction = brain_model.predict(img)
        prediction_image = prediction.reshape((256,256))
        x=(prediction_image*255).astype(np.uint8)
        gray=cv2.Canny(x,50,255)
        contours, hierarchy= cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) ==0:
            y=0
        
    elif(clas=="RETINA"):
        himg=grand.hmap(retina_model,himg)
        prediction = retina_model.predict(img)
        prediction_image = prediction.reshape((256,256))
                
    return prediction_image,y,himg


def intensity_normalization(volume, intensity_clipping_range):
	result = np.copy(volume)

	result[volume < intensity_clipping_range[0]] = intensity_clipping_range[0]
	result[volume > intensity_clipping_range[1]] = intensity_clipping_range[1]

	min_val = np.amin(result)
	max_val = np.amax(result)
	if (max_val - min_val) != 0:
		result = (result - min_val) / (max_val - min_val)

	return result


def predict_nii(img,place):
    try:
        nib_volume = nib.load(img)
        print(type(nib_volume))
    except FileNotFoundError:
        return 0
    
    new_spacing = [1., 1., 1.]
    resampled_volume = resample_to_output(nib_volume, new_spacing, order=1)
    print(type(resampled_volume))
    data = resampled_volume.get_data().astype('float32')
    print(type(data))
    curr_shape = data.shape
    print(curr_shape)

    # resize to get (512, 512) output images
    img_size = 512
    data = zoom(data, [img_size / data.shape[0], img_size / data.shape[1], 1.0], order=1)

    # intensity normalization
    intensity_clipping_range = [-150, 250] # HU clipping limits (Pravdaray's configs)
    data = intensity_normalization(volume=data, intensity_clipping_range=intensity_clipping_range)

    # fix orientation
    data = np.rot90(data, k=1, axes=(0, 1))
    data = np.flip(data, axis=0)

    
    # predict on data
    pred = np.zeros_like(data).astype(np.float32)
    for i in tqdm(range(data.shape[-1]), "pred: "):
        pred[..., i] = liver_model.predict(np.expand_dims(np.expand_dims(np.expand_dims(data[..., i], axis=0), axis=-1), axis=0))[0, ..., 1]
    del data 

    # threshold
    pred = (pred >= 0.4).astype(int)

    # fix orientation back
    pred = np.flip(pred, axis=0)
    pred = np.rot90(pred, k=-1, axes=(0, 1))

    
    # resize back from 512x512
    pred = zoom(pred, [curr_shape[0] / img_size, curr_shape[1] / img_size, 1.0], order=1)
    pred = (pred >= 0.5).astype(np.float32)

    
    # morpological post-processing
    # 1) first erode
    pred = binary_erosion(pred.astype(bool), ball(3)).astype(np.float32)

    # 2) keep only largest connected component
    labels = label(pred)
    regions = regionprops(labels)
    area_sizes = []
    for region in regions:
        area_sizes.append([region.label, region.area])
    area_sizes = np.array(area_sizes)
    tmp = np.zeros_like(pred)
    tmp[labels == area_sizes[np.argmax(area_sizes[:, 1]), 0]] = 1
    pred = tmp.copy()
    del tmp, labels, regions, area_sizes

    # 3) dilate
    pred = binary_dilation(pred.astype(bool), ball(3))

    # 4) remove small holes
    pred = remove_small_holes(pred.astype(bool), area_threshold=0.001*np.prod(pred.shape)).astype(np.float32)

    place.empty()
    place.image('Images/saving.gif')
    pred = pred.astype(np.uint8)
    img = nib.Nifti1Image(pred, affine=resampled_volume.affine)
    resampled_lab = resample_from_to(img, nib_volume, order=0)
    
    return resampled_lab
