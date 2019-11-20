image_mask = Image.open(
            '/storage/UserName/delet_grass_from_build_and_road/db/build/mask_all/' + name_file + '.tiff')
        image_mask = np.array(image_mask)
        self.image_mask_1 = image_mask[:,:,0]
        for j in range(256):
            for k in range(256):
                if image_mask[j,k, 0]!=255 and image_mask[j,k, 1]!=255 and image_mask[j,k, 2]!=255:
                    self.image_mask_1[j,k]=1.0
                else:
                    self.image_mask_1[j,k]=0.0
