dist_matrix=np.ones((height, width))
diagonal=(height**2+width**2)**0.5

for h in range(height):
    for w in range(width):
        if (not(q[h,w]==a[h,w])):
            found=False
            for step_nh in range(height):
                for step_nw in range(width):
                    h_ceiling=(h+step_nh,height-1)[(h+step_nh>=height)]
                    h_floor=(h-step_nh,0)[(h-step_nh)<0]
                    w_left=(w-step_nw,0)[w-step_nh<0]
                    w_right=(w+step_nw,width-1)[((w+step_nw)>=width)]
                    if(not found):
                        for nh in range(h_floor,h_ceiling):
                            for nw in range(w_left,w_right):
                                if(h==nh and w==nw):continue
                                if(q[h,w]==a[nh,nw]):
                                    found=True
                                    dist=((nh-h)**2 + (nw-w)**2)**0.5
                                    if(dist_matrix[h,w]==0 or dist_matrix[h,w]>dist):
                                        dist_matrix[h,w]=dist
                                else:
                                    pass
            if(not found):
                dist_matrix[h,w]=diagonal
