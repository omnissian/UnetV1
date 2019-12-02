#---------------------------------------------------
height=6
width=6
a=np.zeros((height,width))
# q=np.zeros((height,width))
print(a)
for h in range (1,height-2):
    for w in range (1,width-1):
        a[h,w]=1
print("after filling by object")
print("print(a)")
print(a)
q=np.copy(a)
print("print(q)")
print(q)
q[0,1]=1
q[2,2]=0
q[2,3]=0
q[5,1]=1
q[5,0]=0
print("q after addition the mistakes")
print(q)

# distance_matrix=np.ones((height,width))
# print("distance_matrix ", distance_matrix)
# distance_matrix=[[]]
# distance_matrix[0][0].append(1)
#---------------------------------
# distance_matrix[0].append(1)
# distance_matrix[0].append(2)
# distance_matrix.append([1])
# distance_matrix[1].append(2)
# print("distance_matrix ",distance_matrix)
# print("distance_matrix[1,0]= ",distance_matrix[1][0])
# print("distance_matrix[1,0]= ",distance_matrix[0][1])

# distance_matrix.append([[]])

# print("q[2,3]= ",q[2,3])
# print("q[3,4]= ",q[3,4])
dist_matrix=np.ones((height, width))
diagonal=(height**2+width**2)**0.5

for h in range(height):
    for w in range(width):
        print("q[h,w]= ",q[h,w])
        print("a[h,w]= ",a[h,w])
        if (not(q[h,w]==a[h,w])):
            found=False
            for step_nh in range(height):
                for step_nw in range(width):
                    # h_floor=h+step_nh
                    # h_ceiling=h-step_nh
                    # w_left=w-step_nw
                    # w_right=w+step_nw
                    h_low=step_nh+h # cant be higher than height
                    h_top=h-step_nh # cant be lower than zero
                    # iterator for j in range(h_top, h_low) # from lower to higher

                    w_left=w-step_nw # cant be lower than zero
                    w_right=w+step_nw # cant be higher than width
                    # iterator for i in range(w_left, w_right) # from lower to higher obviously

                    h_low=(height-1,step_nh+h)[(step_nh+h)<height]
                    h_top=(0,h-step_nh)[(h-step_nh)>=0]

                    w_left=(0,w-step_nw)[(h-step_nw)>=0]
                    w_right=(width-1,step_nw+w)[(step_nw+w)<width]



                    h_low=(h+step_nh,height-1)[(h+step_nh)>=height]
                    h_top=(h-step_nh,0)[(h-step_nh)<2]

                    w_left=(w-step_nw,0)[(w-step_nw)<2]
                    w_right=(w+step_nw,width-1)[((w+step_nw)>=width)]
                    if(not found):
                        for nh in range(h_top,h_low):
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
print("wtf")

