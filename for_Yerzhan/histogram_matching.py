import nibabel as nib
import matplotlib.pyplot as plt
import os
import numpy as np

path = '../da_img/'

#Taken from open source github
def _match_cumulative_cdf(source, template):
    """
    Return modified source array so that the cumulative density function of
    its values matches the cumulative density function of the template.
    """
    src_values, src_unique_indices, src_counts = np.unique(source.ravel(),
                                                           return_inverse=True,
                                                           return_counts=True)
    tmpl_values, tmpl_counts = np.unique(template.ravel(), return_counts=True)

    # calculate normalized quantiles for each array
    src_quantiles = np.cumsum(src_counts) / source.size
    tmpl_quantiles = np.cumsum(tmpl_counts) / template.size

    interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)
    return interp_a_values[src_unique_indices].reshape(source.shape)


# Loads all images from given domain into list of flat np.arrays
def load_imgs(domain):
    domain_imgs = []

    for file in os.listdir(path):
        if domain in file:
            img = nib.load(path+file)
            data = img.get_fdata()
            flat = data.flatten()
            flat /= np.amax(flat)    # normalizing
            domain_imgs.append(flat)

    return domain_imgs

domain = 'philips_3'
imgs = load_imgs('philips_3')
merged = np.concatenate(imgs)

print(len(np.unique(imgs[0])))     # number of unique elements in the image
print(len(np.unique(imgs[1])))    # then number of unique elements in the merged image
print(len(np.unique(merged)))    # to make sure we concatenate arrays correctly

template = merged
new_path = '../da_img_matched/'

for file in os.listdir(path):
    if domain not in file:
        image = nib.load(path + file)
        data = image.get_fdata()
        matched_image = _match_cumulative_cdf(data, template)

        np.save(os.path.join(new_path, 'matched_'+file.replace('.nii.gz', '')), matched_image)
        print('saved')


# yeah = np.load(new_path+'matched_CC0120_siemens_15_58_F.npy')
#
# print(type(yeah))
# print(yeah.shape)
# #print(np.unique(yeah))
# plt.imshow(yeah[:,:,100])
# plt.show()


