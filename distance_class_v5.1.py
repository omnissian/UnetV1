import numpy as np
import torch.utils.data as data






# #-----------------------------------------------------------
height=6
width=6
label=np.zeros((height,width))
# q=np.zeros((height,width))
print(label)
for h in range (1,height-2):
    for w in range (1,width-1):
        label[h,w]=1
print("after filling by object")
print("print(label)")
print(label)
out=np.copy(label)
print("print(out)")
print(out)
out[0,1]=1
out[2,2]=0
out[2,3]=0
out[5,1]=1
out[5,0]=0
print("out after addition the mistakes")
print(out)
# #-----------------------------------------------------------

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

# dist_matrix=np.ones((height, width))

#---------------------------------
label=np.zeros((height,width))
out=np.ones((height,width))

print("label")
print(label)
print("out")
print(out)


#out=some matrix with width and height
#label=some matrix with same width and height
#unet ++
diagonal=(height**2+width**2)**0.5
dist_matrix=np.full((height,width),diagonal)

radius=( width,height)[(width<height)]

counter=0

#
# for h in range(height):
#     for w in range(width):
#         if(out[h,w]==label[h,w]):
#             counter+=1
#             dist_matrix[h, w] = 0
#             # print("label[",h,",",w,"]= ",label[h,w]," AND ", "out[",h,",",w,"]= ",out[h,w])
#         else:
#             # htop_p=-1
#             # hbot_p=-1
#             # wleft_p=-1
#             # wright_p=-1
#             found=False
#             for r in range(1,radius):
#                 # print("searching inside the LABEL matrix with Radius=",r)
#                 htop=(h-r,0)[(h-r)<0]
#                 hbot=(h+r,(height-1))[(h+r)>=height]
#                 wleft=(w-r,0)[(w-r)<0]
#                 wright=(w+r,(width-1))[(w+r)>=width]
#                 for nh in range(htop,hbot):
#                     counter+=1
#                     # print("label[", nh, ",", wleft, "]= ", label[nh, wleft], " AND ", "out[", h, ",", w, "]= ", out[h, w])
#                     if(label[nh,wleft]==out[h,w]):
#                         # print("THEY ARE EQUAL!!!")
#                         dist=((w-wleft)**2 + (h-nh)**2)**0.5
#                         if(dist_matrix[h,w]>=dist):
#                             dist_matrix[h,w]=dist
#                             found=True
#                             # break
#                     # print("label[", nh, ",", wright, "]= ", label[nh, wright], " AND ", "out[", h, ",", w, "]= ", out[h, w])
#                     if(label[nh,wright]==out[h,w]):
#                         # print("THEY ARE EQUAL!!!")
#                         dist=((w-wright)**2 + (h-nh)**2)**0.5
#                         if(dist_matrix[h,w]>=dist):
#                             dist_matrix[h,w]=dist
#                             found=True
#                             # break
#
#                 for nw in range(wleft,wright):
#                     counter+=1
#                     # print("label[", hbot, ",", nw, "]= ", label[hbot, nw], " AND ", "out[", h, ",", w, "]= ", out[h, w])
#                     if(label[hbot,nw]==out[h,w]):
#                         # print("THEY ARE EQUAL!!!")
#                         dist = ((w - nw) ** 2 + (h - hbot) ** 2) ** 0.5
#                         if(dist_matrix[h,w]>=dist):
#                             dist_matrix[h,w]=dist
#                             found=True
#                     # print("label[", htop, ",", nw, "]= ", label[htop, nw], " AND ", "out[", h, ",", w, "]= ", out[h, w])
#                     if (label[htop, nw] == out[h, w]):
#                         # print("THEY ARE EQUAL!!!")
#                         dist = ((w - nw) ** 2 + (h - htop) ** 2) ** 0.5
#                         if (dist_matrix[h, w] >= dist):
#                             dist_matrix[h, w] = dist
#                             found = True
#                     if(found):
#                         break
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
                # counter+=1
                dist_matrix[h,w]=diagonal

print("dist_matrix")
print(dist_matrix)
print("counter= ",counter)

print("break point")



#-------------------------------------------------------------
# for h in range (height):
#     for w in range(width):
#         if(out[h,w]==label[h,w]):
#             dist_matrix[h, w] = 0
#             counter+=1
#             pass
#         else:
#             found=False
#             for nh in range(height):
#                 for nw in range(width):
#                     counter+=1
#                     if(out[h,w]==label[nh,nw]):
#                         found=True
#                         dist=((h-nh)**2 + (w-nw)**2)**0.5
#                         if(dist_matrix[h,w]>=dist):
#                             dist_matrix[h,w]=dist
#             if(not found):
#                 dist_matrix[h,w]=diagonal
#-------------------------------------------------------------
# for h in range (height):
#     for w in range(width):
#         if(out[h,w]==label[h,w]):
#             dist_matrix[h,w]=0
#             counter+=1
#         else:
#             found=False
#             for r in range(radius):
#                 left=(w-r,0)[(w-r)<0]
#                 right=(w+r,width)[(w+r)>width]
#
#                 top=(h-r,0)[(h-r)<0]
#                 bottom=(h+r,height)[(h+r)>height]
#                 for nh in range (top, bottom):
#                     for nw in range(left,right):
#                         counter += 1
#                         if(nh==h and nw==w):
#                             continue
#                         if(out[h,w]==label[nh,nw]):
#                             dist=((h-nh)**2 + (w-nw)**2)**0.5
#                             if(dist_matrix[h,w]>=dist):
#                                 dist_matrix[h,w]=dist
#                                 found = True
#                 if (found):
#                     break




















# for h in range (height):
#     for w in range(width):
#         if(out[h,w]==label[h,w]):
#             pass
#         else:
#             found=False
#             for nh in range(height):
#                 for nw in range(width):
#                     if(out[h,w]==label[nh,nw]):
#                         found=True
#                         dist=((h-nh)**2 + (w-nw)**2)**0.5
#                         if(dist_matrix[h,w]>=dist):
#                             dist_matrix[h,w]=dist
#             if(not found):
#                 dist_matrix[h,w]=diagonal
#----------------------------------------

#
# for h in range(height):
#     for w in range(width):
#         print("q[h,w]= ",q[h,w])
#         print("a[h,w]= ",a[h,w])
#         if (not(q[h,w]==a[h,w])):
#             found=False
#             for step_nh in range(height):
#                 for step_nw in range(width):
#                     h_low=step_nh+h # cant be higher than height
#                     h_top=h-step_nh # cant be lower than zero
#                     # iterator for j in range(h_top, h_low) # from lower to higher
#
#                     w_left=w-step_nw # cant be lower than zero
#                     w_right=w+step_nw # cant be higher than width
#                     # iterator for i in range(w_left, w_right) # from lower to higher obviously
#
#                     h_low=(height-1,step_nh+h)[(step_nh+h)<height]
#                     h_top=(0,h-step_nh)[(h-step_nh)>=0]
#
#                     w_left=(0,w-step_nw)[(h-step_nw)>=0]
#                     w_right=(width-1,step_nw+w)[(step_nw+w)<width]
#
#
#
#                     h_low=(h+step_nh,height-1)[(h+step_nh)>=height]
#                     h_top=(h-step_nh,0)[(h-step_nh)<2]
#
#                     w_left=(w-step_nw,0)[(w-step_nw)<2]
#                     w_right=(w+step_nw,width-1)[((w+step_nw)>=width)]
#                     if(not found):
#                         for nh in range(h_top,h_low):
#                             for nw in range(w_left,w_right):
#                                 if(h==nh and w==nw):continue
#                                 if(q[h,w]==a[nh,nw]):
#                                     found=True
#                                     dist=((nh-h)**2 + (nw-w)**2)**0.5
#                                     if(dist_matrix[h,w]==0 or dist_matrix[h,w]>dist):
#                                         dist_matrix[h,w]=dist
#                                 else:
#                                     pass
#             if(not found):
#                 dist_matrix[h,w]=diagonal
print("wtf")
