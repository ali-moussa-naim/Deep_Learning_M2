import matplotlib.pyplot as plt
import numpy as np
import csv

# Variable pour les poids et biais
model = np.array([-0.5, 1.2, -1.35])

def initPLT():
    """Initialisation des composants de MatPlotLib"""

    plt.figure(figsize=(20, 12))

    plt.xlabel('x')
    plt.ylabel('y')

    plt.xlim(0, 5.0)
    plt.ylim(0, 5.0)

    plt.grid(alpha=0.5, linewidth = 1)

def drawFigures():
    """Dessin de differentes figures (lignes et points)"""
    plt.plot(1.5, 1.5, marker='o', color='red')
    plt.plot(2.5, 1.5, marker='+', color='green')
    plt.plot(3.5, 1.5, marker='x', color='blue')
    
    plt.scatter(1.5, 2.5, marker='o', color='magenta')
    plt.scatter(2.5, 2.5, marker='+', color='cyan')
    plt.scatter(3.5, 2.5, marker='x', color='purple')
    
    plt.plot([1 , 2],[ 2, 2], color='red', linestyle='solid')
    plt.plot([3 , 4],[ 2, 2], color='green', linestyle='dashed')
    plt.plot([1 , 2],[ 3, 3], color='magenta', linestyle='dashdot')
    plt.plot([3 , 4],[ 3, 3], color='cyan', linestyle='dotted')

    plt.text(1.6, 1.4, 'Label 1', fontsize=9, color='black', rotation=0)
    plt.text(3.6, 1.0, 'Label 2', fontsize=9, color='black', rotation=-30)

# Fonction qui récupère les valeurs du features
def loadfeatures(filename):
    features = np.loadtxt(filename, delimiter=',', comments='#')
    return features

# Fonction qui lit les données du features
def readfeatures(features):
    # Calcul des features avec la fonction z
    l_z = z(model, features)

    # Calcul des predi avec la fonction y
    l_y = y(model, features)

    print("Model")
    print(" w1: ", model[0])
    print(" w2: ", model[1])
    print(" b: ", model[2])

    print("\n")

    dimension = len(features[0])
    size = len(features)

    # Affichage des données
    print("features")
    print(" Dimension: ", dimension)
    print(" Size     : ", size)
    print(" Samples  :")

    # Parcours de toutes les valeurs du features
    if(len(l_z) == len(features)):
        for i in range(len(features)):
            Xk = (features[i][1], features[i][2])
            print("\t" , i , ": " , features[i][0] , "-" , Xk , "-" , "z=" , l_z[i] , "-" , "φ=", phi(l_z[i]) , "-" , "y=" , l_y[i])
    

# Fonction qui affiche sur une figure les données du features
def displayfeatures(features):
    # Réglage de la figure
    plt.figure(figsize=(20,12))

    # Nom des axes
    plt.xlabel('x')
    plt.ylabel('y')

    # Grille de la figure
    plt.grid(alpha=0.1, linewidth=1.0)

    # Parcours des coordonnées dans le sample
    for Xk in features:

        # Affichage du point en rouge si t = 1
        if (Xk[0] == 1):
            plt.plot(Xk[1], Xk[2], marker='o', color='red')
        
        #Affichage du point en vert si t = 0
        else:
            plt.plot(Xk[1], Xk[2], marker='o', color='green')
    
    # Affichage de la figure
    plt.show()

# Fonction qui fait la fonction d'activation
def z(model, features):
    
    # Vérification s'il y a autant de poids que de features
    if(len(model) != len(features[0])):
        print("Il n'y a pas le même nombre de poids que de features.")
        return None
    
    else:
        l_perceptron = []
        for i in range(len(features)):
            z = 0
            for j in range(len(features[0]) - 1):
                z += model[j] * features[i][j + 1]
            
            # Ajout du biais
            z += model[len(model) - 1]
            l_perceptron += [z]
    
    return l_perceptron

# Fonction qui fait la fonction Heaviside
def phi(z):
    if(z >= 0):
        return 1
    else:
        return 0

# Fonction qui calcule la prediction
def y(model, features):
    
    # Vérification s'il y a autant de poids que de features
    if(len(model) != len(features[0])):
        print("Il n'y a pas le même nombre de poids que de features")
        return None
    
    else:
        l_predi = []
        l_tmp = z(model, features)
        for el in l_tmp:
            l_predi.append(phi(el))
        return l_predi
                

def main():
    """Point d'entree du programme
    initPLT()     # Initialisation de MatPlotLib
    #drawFigures() # Dessin des figures
    plt.show()    # Affichage du canevas MatPlotLib
    """

    filename = "datasetA.csv"
    features = loadfeatures(filename)
    readfeatures(features)
    #l_y = y(model, features)
    #print(l_y)


if __name__ == "__main__":
    main()
