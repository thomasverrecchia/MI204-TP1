import numpy as np
import cv2

from matplotlib import pyplot as plt

#Lecture image en niveau de gris et conversion en float64
img=np.float64(cv2.imread('../Image_Pairs/Graffiti0.png',cv2.IMREAD_GRAYSCALE))
(h,w) = img.shape
print("Dimension de l'image :",h,"lignes x",w,"colonnes")
print("Type de l'image :",img.dtype)

#Début du calcul
t1 = cv2.getTickCount()
Theta = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
# Mettre ici le calcul de la fonction d'intérêt de Harris
kernel_dx = np.array([[-1, 0, 1]])
kernel_dy = np.array([[-1, 0, 1]]).T
img3_x = cv2.filter2D(img,-1,kernel_dx)
img3_y = cv2.filter2D(img,-1,kernel_dy)
dx_square = img3_x**2
dy_square = img3_y**2
dx_dy = img3_x * img3_y
alpha = 0.06
W_h = h/100
W_w = w/100
for x in range(h):
    for y in range(w):
        Ksi = np.array([[dx_square[int(x - W_h/2):int(x + W_h/2)+1][int(y - W_w/2):int(y + W_w/2) + 1].sum(), dx_dy[int(x - W_h/2):int(x + W_h/2) + 1][int(y - W_w/2):int(y + W_w/2) + 1].sum()],[dx_dy[int(x - W_h/2):int(x + W_h/2) + 1][int(y - W_w/2):int(y + W_w/2) + 1].sum(), dy_square[int(x - W_h/2):int(x + W_h/2) + 1][int(y - W_w/2):int(y + W_w/2) + 1].sum()]])
        print('sum dx2 vaut : ', dx_square[int(x - W_h/2):int(x + W_h/2)+1][int(y - W_w/2):int(y + W_w/2) + 1].sum(), '\n')
        interest = np.linalg.det(Ksi) - alpha * np.trace(Ksi)
        Theta[x, y] = interest
        print(interest)
# Calcul des maxima locaux et seuillage
Theta_maxloc = cv2.copyMakeBorder(Theta,0,0,0,0,cv2.BORDER_REPLICATE)
d_maxloc = 3
seuil_relatif = 0.01
se = np.ones((d_maxloc,d_maxloc),np.uint8)
Theta_dil = cv2.dilate(Theta,se)
#Suppression des non-maxima-locaux
Theta_maxloc[Theta < Theta_dil] = 0.0
#On néglige également les valeurs trop faibles
Theta_maxloc[Theta < seuil_relatif*Theta.max()] = 0.0
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Mon calcul des points de Harris :",time,"s")
print("Nombre de cycles par pixel :",(t2 - t1)/(h*w),"cpp")

plt.subplot(131)
plt.imshow(img,cmap = 'gray')
plt.title('Image originale')

plt.subplot(132)
plt.imshow(Theta,cmap = 'gray')
plt.title('Fonction de Harris')

se_croix = np.uint8([[1, 0, 0, 0, 1],
[0, 1, 0, 1, 0],[0, 0, 1, 0, 0],
[0, 1, 0, 1, 0],[1, 0, 0, 0, 1]])
Theta_ml_dil = cv2.dilate(Theta_maxloc,se_croix)
#Relecture image pour affichage couleur
Img_pts=cv2.imread('../Image_Pairs/Graffiti0.png',cv2.IMREAD_COLOR)
(h,w,c) = Img_pts.shape
print("Dimension de l'image :",h,"lignes x",w,"colonnes x",c,"canaux")
print("Type de l'image :",Img_pts.dtype)
#On affiche les points (croix) en rouge
Img_pts[Theta_ml_dil > 0] = [255,0,0]
plt.subplot(133)
plt.imshow(Img_pts)
plt.title('Points de Harris')

plt.show()
