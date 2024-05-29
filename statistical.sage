import numpy as np
#from fica import ICA

def Sage2np(MO,n,m):
  MOn=np.matrix(MO)
  return MOn

def runICA(MOn,B=1):
  t1=cputime()
  #A_,S=ICA(MOn,B)
  A_=np.load('./ICA/A_.npy')
  S=np.load('./ICA/S.npy')
  #print('S=',S)
  #print('A=',A_)
  S_=np.dot(np.linalg.inv(A_),MOn)
  print ("time Ica_S: ", cputime(t1)),
  print('S2 before rounding',S2)
  S2=matrix(ZZ,MOn.shape[0],MOn.shape[1], round(S_)) 
  return S2

def runICA_A(MOn,B=1,n=16,kappa=-1):
  t1=cputime()
  #A_,S=ICA(MOn,B,n,kappa)
  A_=np.load('./ICA/A_.npy')
  #S=np.load('./ICA/S.npy')
  #print('S=',S)
  #print('A=',A_)
  print ("time Ica_A: ", cputime(t1)),
  A2=matrix(ZZ,MOn.shape[0],MOn.shape[0],round(A_))
  return A2


def statistical_1(MO,n,m,x0,X,a,b,kappa,B=1,variant=None):  
  
  if variant==None:
    if n<=200: variant='roundA'
    else: variant='roundX'

  print ("Step 2-ICA: ", variant)
  
  
  #print "matNbits=",matNbits(MO),
  tlll=cputime()
  #print('MO before LLL',MO)
  MO=MO.LLL()
  print (" time LLL=",cputime(tlll),"mathNbits=",matNbits(MO)), 

  MOn=Sage2np(MO,n,m)
  #print('MO after LLL',MO)
  #print('MOn',MOn)
  np.save('./ICA/MOn.npy',MOn)
  #print("type MOn")
  #print(type(MOn))
  return MOn,MO

def statistical_2(MOn,MO,n,m,x0,X,a,b,kappa,B=1,variant=None):  
  t2=cputime()
  if variant==None:
    if n<=200: variant='roundA'
    else: variant='roundX'
  if variant=="roundA":
    A2=runICA_A(MOn,B,n,kappa)
    try:
      S2 = A2.inverse() * MO
      print("mathNbits A=", matNbits(A2)),
    except:
      return 0,0,0
  elif variant=="roundX":
    S2=runICA(MOn,B)
    print ("mathNbits X=",matNbits(X)),
  else:
    raise NameError('Variant algorithm non acceptable') 
  tica=cputime(t2)
  print( " cputime ICA %.2f" %tica),  

  tc=cputime()
  Y=X.T
  #print(Y)
  nfound=0
  for i in range(n):
    for j in range(n):
      if S2[i,:n]==Y[j,:n] and S2[i]==Y[j]:  
        nfound+=1
  t=cputime(tc)      
  print( "  NFound=",nfound,"out of",n,"check= %.2f" %t)

#  print('S2=',S2)
#  print('Y=',Y)
#  if nfound<n: 
#     return tica, 0 ,nfound

  NS=S2.T
  resX=np.save('./ICA/resX.npy',S2)

  
  tcoff=cputime()
  #b=X*a=NS*ra
  invNSn=matrix(Integers(x0),NS[:n]).inverse()
  ra=invNSn*b[:n]
  #print('Real a=',a)
  #print('Reconstructed a=',ra)
  tcf= cputime(tcoff)

  nrafound=len([True for rai in ra if rai in a])   
  print( "  Coefs of a found=",nrafound,"out of",n, " time= %.2f" %tcf)
  
  tS2=tcf+tica
  print( "  Total step2: %.1f" % tS2),
  
  return tica, tS2, nrafound


def statistical_2_mat(MOn, MO, n, m, x0, X, a, b, kappa, B=1, variant=None):
  print("statistical_2_mat here")
  t2 = cputime()
  if variant == None:
    if n <= 200:
      variant = 'roundA'
    else:
      variant = 'roundX'
  if variant == "roundA":
    A2 = runICA_A(MOn, B, n, kappa)
    print(A2)
    try:
      S2 = A2.inverse() * MO
      print("mathNbits A=", matNbits(A2)),
    except:
      return 0, 0, 0,0,0,0
  elif variant == "roundX":
    S2 = runICA(MOn, B)
    print("mathNbits X=", matNbits(X)),
  else:
    raise NameError('Variant algorithm non acceptable')
  tica = cputime(t2)
  print(" cputime ICA %.2f" % tica),

  tc = cputime()
  Y = X.T
  #print(Y)
  nfound = 0
  for i in range(n):
    for j in range(n):
      if S2[i, :n] == Y[j, :n] and S2[i] == Y[j]:
        nfound += 1
  t = cputime(tc)
  print("  NFound=", nfound, "out of", n, "check= %.2f" % t)

  #  print('S2=',S2)
  #  print('Y=',Y)
  #  if nfound<n:
  #     return tica, 0 ,nfound

  NS = S2.T
  resX = np.save('./ICA/resX.npy', S2)

  tcoff = cputime()
  # b=X*a=NS*ra
  invNSn = matrix(Integers(x0), NS[:n]).inverse()
  ra = invNSn * b[:n]
  # print('Real a=',a)
  # print('Reconstructed a=',ra)
  tcf = cputime(tcoff)

  nrafound = len([True for rai in ra if rai in a])
  print("  Coefs of a found=", nrafound, "out of", n, " time= %.2f" % tcf)

  tS2 = tcf + tica
  print("  Total step2: %.1f" % tS2),

  return tica, tS2, nrafound, NS.T,nfound,nrafound
