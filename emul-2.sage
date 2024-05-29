load("hssp.sage")
load("extendhssp.sage")
load("multi-dim-hssp.sage")
import numpy as np
import random

print "---Emulatig uniform distribution---"
n=10
l=10
m0=100
data=np.load("del_X_fl.npy")
D=matrix(ZZ,data)
e=vector(ones_matrix(1,n))
freqD=(n+1)*[0]
for x in e*D: freqD[x]+=1
freqU=[binomial(n,i)/2.**n for i in range(n+1)]
c=dict()
for x in D.T: 
   h=x*e
   if h in c.keys(): c[h]+=[x]
   else: c[h]=[x]

sampleSize=[int(m0*f) for f in freqU]
L=[]
for i in range(n+1):
      L+=random.sample(c[i], k=sampleSize[i])

while True:
   random.shuffle(L)
   newX=matrix(ZZ,L)
   if newX.rank()==n: break
   
freqnewX=(n+1)*[0]
for x in newX*e: freqnewX[x]+=1
G1=bar_chart(freqD, legend_label='frequency hw input')
G2=bar_chart(freqnewX, legend_label='frequency hw output')
G3=bar_chart(freqU, legend_label='frequency hw uniform')
#for g in [G1,G2,G3]: g.show()


print "---Random A---"
m=(newX.dimensions()[0])//n*n
H=multi_hssp(n,l)
H.gen_instance(m=m)
#x0,A,dummyX,dummyB=genParams_mat(m=m,l=10)
H.X=newX[:m]
x0=H.x0
Y=newX[:m]
A=H.A
B=(Y*H.A)%H.x0
H.B=(H.X*H.A)%H.x0
print"\n--local"
ke,tt1,tt10,tt1O=Step1_Mat_loc(n,x0,A,Y,B,m)
MB,beta,rA =Step2_BK_mat_loc(ke,Y,A,n,m,x0,B)
print"\n--class"
hssp_attack_multi(H,'ns_original')

print
print "\n---images data A---"
alp=np.load("alpha10_3072.npy")
l=10
Aifull=matrix(ZZ,alp)
someA=random.sample(Aifull.columns(),l)
Ai=matrix(ZZ,someA).T
H.A=Ai
H.B=(H.X*H.A)%H.x0
Bi=Y*Ai%x0
print"\n--local"
kei,tt1,tt10,tt1O=Step1_Mat_loc(n,x0,Ai,Y,Bi,m)
MBi,beta,rAi =Step2_BK_mat_loc(kei,Y,Ai,n,m,x0,Bi)

print"\n--class"
hssp_attack_multi(H,'ns_original')

"""
print
print "\n---images data A case 2---"
alp2=np.load("alpha10_784.npy")
l=10
Aifull2=matrix(ZZ,alp2)
someA2=random.sample([a for a in Aifull2.columns() if a!=0],l)
Ai2=matrix(ZZ,someA2).T
H.A=Ai2
H.B=(H.X*H.A)%H.x0
Bi2=Y*Ai2%x0
kei2,tt1,tt10,tt1O=Step1_Mat(n,x0,Ai2,Y,Bi2,m)

print"\n--local"
Ai=matrix(ZZ,someA).T
H.A=Ai
H.B=(H.X*H.A)%H.x0
Bi=Y*Ai%x0
print"\n--class"
MBi2,beta,rAi2 =Step2_BK_mat(kei2,Y,Ai2,n,m,x0,Bi2)
"""
