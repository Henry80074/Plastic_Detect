    # mask[mask == 15] = 7  # Mixed Water to Marine Water Class
    # mask[mask == 14] = 7  # Wakes to Marine Water Class
    # mask[mask == 13] = 7  # Cloud Shadows to Marine Water Class
    # mask[mask == 12] = 7  # Waves to Marine Water Class

    # reduce every value by 1
    # mask = np.copy(mask - 1)
    # mask[mask < 0] = nan

    # code start, making int mask and tiling to match image dims
    # mask_bool = mask == 0 #-may not be needed
    # mask_int = mask_bool.astype(int)
    # mask_cop = np.tile(mask_int, (11, 1, 1))
    # end code #

     # FOR CROPPING AND RESIZING - duplicate previous commented code if needed

    #Extract patches from each image - if image bigger than 256x256
    # print("Now patchifying image:", dataset_path + "/" + roi_name)
    # patches_img = patchify(image, (11, patch_size, patch_size),
    #                        step=patch_size)  # Step=256 for 256 patches means no overlap
    #
    # for i in range(patches_img.shape[0]):
    #     for j in range(patches_img.shape[1]):
    #         single_patch_img = patches_img[i, j, :, :]
    #
    #         # Use minmaxscaler instead of just dividing by 255.
    #         single_patch_img = scaler.fit_transform(
    #             single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(
    #             single_patch_img.shape)
    #
    #         # single_patch_img = (single_patch_img.astype('float32')) / 255.
    #         single_patch_img = single_patch_img[0]  # Drop the extra unecessary dimension that patchify adds.
    #         image_dataset.append(single_patch_img)
