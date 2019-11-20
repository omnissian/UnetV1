
#-------------found number of all classes-----------------
original_pixels=[]
adds=0
#---test!!! adds=1
adds=1

#---test!!! adds=1
for iter_valid_data, (input_valid, targets_valid) in enumerate(data_valid_loader):
    original_pixels=(targets_valid.permute(1,0,2,3)).data[0][:,0,0]

    # ---test!!! adds=1
    test_cat=original_pixels
    # test_cat=torch.tensor([0,0,0],device='cpu')
    print("test_cat.size() ",test_cat.size())
    original_pixels=torch.stack([original_pixels,original_pixels],dim=0)
    print("original_pixels.size() ",original_pixels.size())
    # original_pixels=torch.cat([original_pixels.data[0],test_cat],dim=0)
    # torch.cat([original_pixels,original_pixels],dim=0]
    print("original_pixels.size() ",original_pixels.size())
    print("len(original_pixels) ",len(original_pixels))
    # ---test!!! adds=1
    tmp = (targets_valid.permute(1, 0, 2, 3)).data[0]
    channels = targets_valid.permute(1, 0, 2, 3).size()
    width = channels[3]
    height = channels[2]
    channels = channels[1]
    # wtf1=targets_valid.permute(1,0,2,3)
    print("targets_valid.size() ",targets_valid.size())
    for i in range(batch_size):
        # original_pix=np.zeros((width,height,channels))
        # original_pixels=torch.tensor([0,0,0]) # for rgb
        for i in range(width):
            for j in range(height):
                pix=tmp.data[:,i,j]
                WTF_len=len(original_pixels)
                if(adds):
                    for ip in range (len(original_pixels)):
                        # dict_cur_pix=original_pixels.data[0]
                        left=original_pixels.data[ip]
                        right=pix.data
                        print(right)
                        print(left)
                        wtf_logic = (torch.all(torch.eq(pix, original_pixels.data[ip]))).item()
                        if(wtf_logic):
                            break
                            pass
                        else:
                            right=right.unsqueeze(0)
                            print("left= ",left)
                            print("right= ",right)

                            # original_pixels=torch.cat([original_pixels,pix],dim=0)
                            original_pixels=torch.cat([original_pixels,right],dim=0)
                            break

                else:
                    wtf_logic=(torch.all(torch.eq(pix,original_pixels))).item()
                    if(wtf_logic):
                        pass
                    else:
                        torch.stack([original_pixels,pix],dim=0)
                        adds+=1;

                    # wtf_logic=(torch.all(torch.eq(pix,original_pixels))).data[0]

                # print("original_pixels.size() ",original_pixels.size())
                # print("original_pixels[0].size() ",original_pixels[0].size())
                # torch.eq(pix,)

            pixel_ch = 0
            img_out=tmp.data[ch]
            width,height =img_out.size()
            for i in range(width):
                for j in range (height):
                    img_out
        #----------------------------------------------------
        # for ch in range(channels):
        #     pixel_ch = 0
        #     img_out=tmp.data[ch]
        #     width,height =img_out.size()
        #     for i in range(width):
        #         for j in range (height):
        #             img_out

# -------------found number of all classes-----------------
