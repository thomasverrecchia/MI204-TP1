# Compte Rendu TP1 : Détection et Appariement de Points Caractéristiques

## 2 Formats d'images et convolutions

1. Expérimentation du code. 


2. Le noyau de convolution choisit permet de réaliser 
un réhausement de contraste car on multiplie la valeur
du pixel par 5 et on soustrait les valeurs des pixels autour.
La somme des coefficient vaut un donc en moyenne le niveau de 
gris reste identique, cependant la valeur du pixel considéré 
étant positive et celles des pixels autour étant négatives
cela permet de réhausser le contraste.


3. Pour avoir un affichage correct, il faut s'assurer de
prendre un vmin = -128 et un vmax = 128 dans la fonction
imshow().
```python
t1 = cv2.getTickCount()
kernel = np.array([[0, -1, 0],[-1, 5, -1],[0, -1, 0]])
kernel_dx = np.array([[-1,0,1]])
kernel_dy = kernel_dx.T
img3 = cv2.filter2D(img,-1,kernel)
img_3_dx = cv2.filter2D(img,-1,kernel_dx)
img_3_dy = cv2.filter2D(img,-1,kernel_dy)

cv2.imshow('Avec filter2D',img3/255.0)
cv2.waitKey(0)
plt.subplot(122)
plt.imshow(img_3_dx,cmap = 'gray',vmin = -128.0,vmax = 128.0)
```

Pour calculer la norme euclidienne du gradient de l'image on a rajouter la ligne de code suivante: 

    norm_img = np.sqrt(img_3_x**2 + img_3_y**2)

Pour l'affichage on a réutiliser vmin = 0 et vmax = 255 car on ne peut pas avori de norme négative.

![Partie 2](Part2.png)

## 3 Détecteurs

4. mes couilles

            
