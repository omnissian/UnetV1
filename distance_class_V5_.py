
#out=some matrix with width and height
#label=some matrix with same width and height
#unet ++
diagonal=(height**2+width**2)**0.5
dist_matrix=np.full((height,width),diagonal)

radius=( width,height)[(width<height)]

counter=0


for h in range(height):
    for w in range(width):
        if(out[h,w]==label[h,w]):
            counter+=1
            dist_matrix[h, w] = 0
            # print("label[",h,",",w,"]= ",label[h,w]," AND ", "out[",h,",",w,"]= ",out[h,w])
        else:
            # htop_p=-1
            # hbot_p=-1
            # wleft_p=-1
            # wright_p=-1
            found=False
            for r in range(1,radius):
                # print("searching inside the LABEL matrix with Radius=",r)
                htop=(h-r,0)[(h-r)<0]
                hbot=(h+r,(height-1))[(h+r)>=height]
                wleft=(w-r,0)[(w-r)<0]
                wright=(w+r,(width-1))[(w+r)>=width]
                for nh in range(htop,hbot):
                    counter+=1
                    # print("label[", nh, ",", wleft, "]= ", label[nh, wleft], " AND ", "out[", h, ",", w, "]= ", out[h, w])
                    if(label[nh,wleft]==out[h,w]):
                        # print("THEY ARE EQUAL!!!")
                        dist=((w-wleft)**2 + (h-nh)**2)**0.5
                        if(dist_matrix[h,w]>=dist):
                            dist_matrix[h,w]=dist
                            found=True
                            # break
                    # print("label[", nh, ",", wright, "]= ", label[nh, wright], " AND ", "out[", h, ",", w, "]= ", out[h, w])
                    if(label[nh,wright]==out[h,w]):
                        # print("THEY ARE EQUAL!!!")
                        dist=((w-wright)**2 + (h-nh)**2)**0.5
                        if(dist_matrix[h,w]>=dist):
                            dist_matrix[h,w]=dist
                            found=True
                            # break

                for nw in range(wleft,wright):
                    counter+=1
                    # print("label[", hbot, ",", nw, "]= ", label[hbot, nw], " AND ", "out[", h, ",", w, "]= ", out[h, w])
                    if(label[hbot,nw]==out[h,w]):
                        # print("THEY ARE EQUAL!!!")
                        dist = ((w - nw) ** 2 + (h - hbot) ** 2) ** 0.5
                        if(dist_matrix[h,w]>=dist):
                            dist_matrix[h,w]=dist
                            found=True
                    # print("label[", htop, ",", nw, "]= ", label[htop, nw], " AND ", "out[", h, ",", w, "]= ", out[h, w])
                    if (label[htop, nw] == out[h, w]):
                        # print("THEY ARE EQUAL!!!")
                        dist = ((w - nw) ** 2 + (h - htop) ** 2) ** 0.5
                        if (dist_matrix[h, w] >= dist):
                            dist_matrix[h, w] = dist
                            found = True
                    if(found):
                        break
#--------------------------------

for h in range (height):
    for w in range(width):
        if(out[h,w]==label[h,w]):
            dist_matrix[h, w] = 0
            counter+=1
            pass
        else:
            found=False
            for nh in range(height):
                for nw in range(width):
                    counter+=1
                    if(out[h,w]==label[nh,nw]):
                        found=True
                        dist=((h-nh)**2 + (w-nw)**2)**0.5
                        if(dist_matrix[h,w]>=dist):
                            dist_matrix[h,w]=dist
            if(not found):
                counter+=1
                dist_matrix[h,w]=diagonal

print("dist_matrix")
print(dist_matrix)
print("counter= ",counter)

print("break point")
