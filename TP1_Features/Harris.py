import numpy as np
import cv2

from matplotlib import pyplot as plt
def harris(reduction, alpha):
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
    W_h = h/reduction
    W_w = w/reduction
    for x in range(h):
        for y in range(w):
            sub_dxdx = dx_square[int(x - W_h/2):int(x + W_h/2)+1,int(y - W_w/2):int(y + W_w/2) + 1]
            sub_dxdy = dx_dy[int(x - W_h/2):int(x + W_h/2) + 1, int(y - W_w/2):int(y + W_w/2) + 1]
            sub_dydy = dy_square[int(x - W_h/2):int(x + W_h/2) + 1 , int(y - W_w/2):int(y + W_w/2) + 1]
            Ksi = np.array([[np.sum(sub_dxdx), np.sum(sub_dxdy)],[np.sum(sub_dxdx), np.sum(sub_dydy)]])
            interest = np.linalg.det(Ksi) - alpha * np.trace(Ksi)**2
            Theta[x, y] = interest

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

    '''plt.subplot(131)
    plt.imshow(img,cmap = 'gray')
    plt.title('Image originale')
    
    plt.subplot(132)
    plt.imshow(Theta,cmap = 'gray')
    plt.title('Fonction de Harris')'''

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
    return Img_pts
'''    plt.subplot(133)
    plt.imshow(Img_pts)
    plt.title('Points de Harris')
    
    plt.show()
'''
img100_006 = harris(100, 0.06)
img1000_006 = harris(1000, 0.06)
img10_006 = harris(10, 0.06)

img100_06 = harris(100, 0.6)
img100_00 = harris(100, 0.0)

plt.subplot(131)
plt.imshow(img100_00)
plt.title(r'$\alpha$ = 0')

plt.subplot(132)
plt.imshow(img100_006)
plt.title(r'$\alpha$ = 0.06')

plt.subplot(133)
plt.imshow(img100_06)
plt.title(r'$\alpha$ = 0.6')

plt.show()


