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
b=np.copy(a)
print("print(b)")
print(b)
b[0,1]=1
b[2,2]=0
b[2,3]=0
b[5,1]=1
b[5,0]=0
print("b after addition the mistakes")
print(b)



print("break point")

#---------------------------------------------------
