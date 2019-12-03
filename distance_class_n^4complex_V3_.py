#out=some matrix with width and height
#label=some matrix with same width and height

diagonal=(height**2+width**2)**0.5
dist_matrix=np.full((height,width),diagonal)


for h in range (height):
    for w in range(width):
        if(out[h,w]==label[h,w]):
            dist_matrix[h,w]=0
        else:
            for nh in range(height):
                for nw in range(width):
                    if(out[h,w]==label[nh,nw]):
                        found=True
                        dist=((h-nh)**2 + (w-nw)**2)**0.5
                        if(dist_matrix[h,w]>=dist):
                            dist_matrix[h,w]=dist


print("dist_matrix")
print(dist_matrix)
