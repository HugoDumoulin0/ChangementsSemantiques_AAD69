# Analyse diachronique du changement sémantique à travers trois modèles distributionnels

Projet Tuteuré 2026 master Plurital M2

Hugo Dumoulin – hdumoulin@parisnanterre.fr

## Problématique scientifique

Les modèles distributionnels constituent un outil séduisant pour l’étude du changement linguistique (Hamilton, Leskovec & Jurasfky 2016). Cependant, ils reposent sur des fondations théoriques et méthodologiques très diverses : modèles neuronaux prédictifs, matrices statistiquement pondérées, ou encore dispositifs issus de la tradition de l’analyse du discours. Comparer de manière systématique leurs capacités à représenter les évolutions sémantiques dans le temps permet de rendre compte de différences d’interprétation, de granularité et de robustesse. 
Ce projet propose de comparer trois modèles : (1) les modèles neuronaux de type Word2Vec ou FastText (Mikolov et al. 2013), (2) les matrices de cooccurrences généralisées (ou DSM) telles que formalisées dans les travaux de Stefanie Evert (2005), et (3) le dispositif AAD69 , approche structurale et discursive centrée sur la notion d’énoncé et sur la construction de domaines sémantiques (Pêcheux 1969). L’objectif est d’étudier leurs convergences, leurs divergences, et leur capacité respective à rendre compte de changements sémantiques réels observables dans un corpus de référence.
Ces trois approches seront confrontées à un même corpus de presse diachronique constitué à partir des trois journaux Le Figaro, La Croix et L’Humanité, depuis leur création jusqu’à 1954, qui sera divisé en périodes. Les documents sont accessibles sur Gallica grâce à l’API Document. 

## Travaux à réaliser

Le projet se décompose en plusieurs étapes complémentaires :

1. Mise en œuvre de l’algorithme AAD69  
Sur la base du travail de l’année précédente, les étudiants reconstruiront en Python l’algorithme décrit par Pêcheux dans l’ouvrage de 1969 et ses prolongements. Le dispositif doit réaliser : 
– le découpage du corpus en énoncés élémentaires reliés entre eux par des relations syntaxiques et discursives 
– la construction de matrices lexicales énoncés × mots ;  
– le calcul de proximités entre énoncés ;  
– la projection factorielle ou la clusterisation permettant d’identifier les domaines sémantiques.

2. Construction de modèles DSM 
À partir du même corpus, les étudiants construiront une matrice de cooccurrences généralisée, appliqueront des pondérations, puis éventuellement une réduction de dimension (SVD). Ils étudieront la stabilité, l’interprétabilité et la capacité du DSM à révéler des évolutions sémantiques.

3. Entraînement de modèles Word2Vec/FastText  
Un modèle prédictif sera entraîné pour chaque période du corpus. Les espaces vectoriels seront alignés afin de suivre la trajectoire sémantique de mots cibles. Les dérives sémantiques seront mesurées via distance cosinus, changement de voisinage, ou réduction dimensionnelle.

4. Analyse comparative des trois approches  
Les étudiants compareront les résultats produits par les trois méthodes sur un ensemble de mots sélectionnés (politiques, techniques, culturels). Ils analyseront :  
– les types de changements capturés ;  
– la finesse des distinctions sémantiques ;  
– l’interprétabilité des résultats ;  
– les effets de corpus et les limites méthodologiques.  
L’articulation entre analyse lexicale, analyse thématique et analyse discursive constituera un axe majeur de la réflexion.

## Livrables attendus

– Une reconstruction fonctionnelle en Python de l’algorithme AAD69.  
– Les modèles DSM et Word2Vec entraînés par période, accompagnés de leurs analyses.  
– Un rapport final comprenant une analyse linguistique et scientifique approfondie et une comparaison détaillée des trois approches.  
– Des visualisations (graphiques, cartes sémantiques, trajectoires temporelles) illustrant les évolutions observées sur les corpus diachroniques
Ce projet tuteuré permettra ainsi aux étudiants d’acquérir une compréhension fine des différents paradigmes de la sémantique distributionnelle, tout en mobilisant des compétences pratiques en ingénierie linguistique, en programmation et en analyse statistique.

## Supports

Github du projet : https://github.com/HugoDumoulin0/ChangementsSemantiques_AAD69 (me contacter pour l’accès) 
Gihub du projet 2025 : https://github.com/lucilebessac/AAD69-synonymie-contextuelle

## Bibliographie

Evert, S. (2005). The Statistics of Word cooccurrences :Word pairs and Collocations. Phd Thesis. University of Stuttgart.
Hamilton, W. L., Leskovec, J., & Jurafsky, D. (2016). “Diachronic Word Embeddings Reveal Statistical Laws of Semantic Change”. Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 1489‑1501. 
Léon, J. (2010). « AAD69 : Archéologie d’une étrange machine. » Semen, 29, 89 90.
Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient Estimation of Word Representations in Vector Space.
Pêcheux, M. (1969). Analyse automatique du discours. Dunod.
