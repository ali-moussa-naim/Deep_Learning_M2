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

    # On récupère l'erreur
    e = error(features, model)

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

        print(" Erreur:", e)
    else:
        print("il faut qu'il y ait le même nombre de sorties que de données")
        return 
    

# Fonction qui affiche sur une figure les données du features
def displayDataset(features, model):
    # Liste des predictions
    l_predi = y(model, features)
    
    # On vérifie s'il y a le même nombre de classes que predi
    if(len(features) != len(l_predi)):
        print("Il n'y a pas le même nombre de classes que de predi")
        return
    

    # Réglage de la figure
    plt.figure(figsize=(20,12))

    # Nom des axes
    plt.xlabel('x')
    plt.ylabel('y')

    plt.xlim(0, 5.0)
    plt.ylim(0, 5.0)

    # Grille de la figure
    plt.grid(alpha=0.1, linewidth=1.0)

    # Parcours des coordonnées dans le sample
    for i in range(len(features)):

        #Affichage du point en vert si classe = 0 et predi = 0
        if(features[i][0] == 0 and l_predi[i] == 0):
            plt.plot(features[i][1], features[i][2], marker='o', color='green')
        
        # Affichage du point en cyan si classe = 0 et predi = 1
        elif(features[i][0] == 0 and l_predi[i] == 1):
            plt.plot(features[i][1], features[i][2], marker='o', color='cyan')

        # Affichage du point en rouge si t = 1 et predi = 1
        elif(features[i][0] == 1 and l_predi[i] == 1):
            plt.plot(features[i][1], features[i][2], marker='o', color='red')
        
        # Affichage du point en magenta si t = 1 et predi = 0
        elif(features[i][0] == 1 and l_predi[i] == 0):
            plt.plot(features[i][1], features[i][2], marker='o', color='magenta')
        
        #Affichage du point en gris si pas de prédiction associée
        else:
            plt.plot(features[i][1], features[i][2], marker='o', color='gray')
    
    # Affichage ligne frontière décision
    displayFrontier(model)

    # Affichage de l'erreur
    e = error(features, model)
    plt.text(0, 4.9, f'Error:{e}', fontsize=9, color='black', rotation=0)
    
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

# Fonction qui affiche la frontière de décision
def displayFrontier(model):
    w1, w2, b = model
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()

    if w2 != 0:
        x1s = [xmin, xmax]
        x2s = [(-w1 * xmin - b) / w2, (-w1 * xmax - b) / w2]
    elif w1 != 0:
        x1s = [-b / w1, -b / w1]
        x2s = [ymin, ymax]
    else:
        return

    plt.plot(x1s, x2s, color='orange', linewidth=2)

# Fonction qui calcule et retourne l'erreur e
def error(features, model):

    # Ajout des classes de tous les individus
    l_classe = []
    for Xk in features:
        l_classe.append(Xk[0])
    
    # On récupère toutes les prédi
    l_predi = y(model, features)

    # On vérifie qu'il y a autant de prédi que de classes
    if(len(l_classe) != len(l_predi)):
        print("Il faut qu'il y ait le même nombre de predi que de classes")
        return

    e = 0
    for i in range(len(features)):
        e += (l_classe[i] - l_predi[i])
    
    return e



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
    displayDataset(features, model)

if __name__ == "__main__":
    main()
