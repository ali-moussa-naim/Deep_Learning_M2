import matplotlib.pyplot as plt
import numpy as np

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


def 

def main():
    """Point d'entree du programme"""
    initPLT()     # Initialisation de MatPlotLib
    #drawFigures() # Dessin des figures
    plt.show()    # Affichage du canevas MatPlotLib

if __name__ == "__main__":
    main()
