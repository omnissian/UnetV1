#out=some matrix with width and height
#label=some matrix with same width and height

diagonal=(height**2+width**2)**0.5
dist_matrix=np.full((height,width),diagonal)

radius=( width,height)[(width<height)]

for h in range (height):
    for w in range(width):
        if(out[h,w]==label[h,w]):
            dist_matrix[h,w]=0
        else:
            found=False
            for r in range(radius):
                left=(w-r,0)[(w-r)<0]
                right=(w+r,width)[(w+r)>width]

                top=(h-r,0)[(h-r)<0]
                bottom=(h+r,height)[(h+r)>height]
                for nh in range (top, bottom):
                    for nw in range(left,right):
                        if(out[h,w]==label[nh,nw]):
                            dist=((h-nh)**2 + (w-nw)**2)**0.5
                            if(dist_matrix[h,w]>=dist):
                                dist_matrix[h,w]=dist
                                found = True
                if (found):
                    break
