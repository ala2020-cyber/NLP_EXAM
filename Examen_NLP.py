import spacy
import math
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import os

os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Remplace 4 par le nombre de cœurs logiques de ton processeur


# Configuration du logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Charger le modèle spaCy
nlp = spacy.load("fr_core_news_sm")



# Définition des corpus
corpus_1 = [
    "L'intelligence artificielle (IA) est un domaine vaste qui inclut un ensemble de techniques et de théories permettant de créer des systèmes capables de simuler des processus cognitifs humains. Cela inclut l'apprentissage, la prise de décision, la reconnaissance des émotions, et même l'interaction avec l'environnement. Dans ce contexte, l'IA s'applique à des secteurs très diversifiés, allant de la santé à l'éducation, en passant par la finance et la robotique.",
    "Les réseaux neuronaux artificiels sont l'une des techniques les plus populaires dans le domaine de l'IA. Inspirés du fonctionnement du cerveau humain, ces réseaux sont utilisés pour résoudre des problèmes complexes comme la reconnaissance d'image, la prédiction de données ou l'apprentissage automatique. Ils sont au cœur de nombreuses innovations actuelles telles que la conduite autonome et l'analyse de données massives.",
    "Les applications de l'IA sont omniprésentes dans notre quotidien. Par exemple, les moteurs de recherche utilisent des algorithmes d'IA pour fournir des résultats plus pertinents. L'IA est également présente dans les systèmes de recommandation de plateformes comme Netflix ou Amazon, où elle aide à prédire les préférences des utilisateurs. Dans le domaine médical, l'IA est utilisée pour analyser des images médicales et aider les médecins à diagnostiquer des maladies avec une précision accrue.",
    "Les véhicules autonomes représentent une des applications les plus fascinantes de l'IA. Ces véhicules utilisent une combinaison de capteurs, d'algorithmes d'IA et de réseaux neuronaux pour percevoir leur environnement et prendre des décisions en temps réel. La technologie s'améliore rapidement, et certaines entreprises comme Tesla, Waymo et Uber investissent massivement dans la recherche pour rendre ces véhicules totalement autonomes.",
    "L'intelligence artificielle dans la finance a permis des avancées majeures. Les banques et les compagnies d'assurance utilisent l'IA pour analyser les tendances du marché, prédire les comportements des consommateurs et même détecter les fraudes. Les systèmes de trading automatisés basés sur l'IA sont devenus des acteurs incontournables dans le monde de la finance, réalisant des transactions en une fraction de seconde avec des algorithmes qui surpassent les capacités humaines en termes de vitesse et de précision.",
    "La révolution de l'IA n'est pas seulement technologique, elle soulève aussi de nombreuses questions éthiques. Par exemple, l'utilisation des données personnelles pour entraîner des modèles d'IA soulève des préoccupations concernant la vie privée et la sécurité. De plus, la question de l'impact de l'automatisation sur l'emploi est un sujet de débat crucial, avec des craintes de perte d'emplois dans certains secteurs en raison de l'introduction de l'IA."
]

corpus_3 = [
    "L'intelligence artificielle (IA) désigne un ensemble de méthodes avancées permettant aux machines de reproduire des tâches cognitives complexes qui étaient autrefois réservées aux humains. Cette capacité à apprendre et à s'adapter à des situations nouvelles permet à l'IA de trouver des solutions à des problèmes de plus en plus complexes, et d'appliquer ces solutions à des domaines variés tels que la logistique, les transports et l'éducation.",
    "Parmi les nombreuses technologies sous-jacentes à l'IA, les réseaux neuronaux artificiels jouent un rôle crucial. Ces réseaux, composés de couches multiples de neurones simulés, sont capables de traiter des informations de manière hiérarchique, ce qui leur permet d'effectuer des tâches comme la reconnaissance vocale, l'analyse d'images ou la traduction automatique. L'apprentissage profond, une méthode d'IA qui repose sur l'utilisation de réseaux neuronaux complexes, est à l'origine de nombreuses percées technologiques récentes.",
    "Les applications de l'IA ne cessent de se diversifier. Outre les moteurs de recherche et les systèmes de recommandation, l'IA est de plus en plus utilisée dans la finance pour prédire les fluctuations des marchés boursiers, ou dans le domaine de la santé pour proposer des traitements personnalisés. Toutefois, ces avancées soulèvent des questions éthiques importantes, notamment concernant la vie privée des utilisateurs et les décisions automatisées prises par des algorithmes sans intervention humaine.",
    "L'un des domaines émergents dans l'application de l'IA est l'industrie créative. Les artistes, designers et réalisateurs de films commencent à utiliser l'IA pour générer des œuvres originales, de la musique à la peinture. Cette utilisation de l'IA soulève des questions sur la créativité et l'originalité, et sur la place des artistes humains dans un monde où les machines peuvent produire des œuvres d'art autonomes.",
    "L'intelligence artificielle est également utilisée dans le domaine de la cybersécurité pour détecter des menaces et des attaques potentielles en temps réel. Les systèmes basés sur l'IA peuvent analyser des volumes massifs de données pour identifier des comportements suspects, prédire des attaques et même défendre les systèmes sans intervention humaine. Cependant, cette capacité à anticiper les menaces pose également des défis liés à la sécurité des données et aux risques de manipulation de ces systèmes.",
    "Un autre domaine fascinant de l'IA est l'éthique de l'intelligence artificielle. Alors que la technologie progresse, des questions cruciales se posent sur la manière de garantir que les systèmes d'IA prennent des décisions éthiques et ne reproduisent pas des biais humains. Par exemple, les biais de genre ou de race dans les modèles d'IA peuvent avoir des conséquences graves, notamment dans des domaines comme le recrutement, la justice pénale ou la surveillance."
]



corpus_2 = [
    "L'économie mondiale est influencée par de nombreux facteurs, notamment la politique monétaire, les fluctuations des marchés financiers et les échanges commerciaux internationaux. Les gouvernements et les institutions financières utilisent divers outils pour stabiliser l'économie et encourager la croissance, tels que les taux d'intérêt, les politiques budgétaires et les régulations bancaires.",
    
    "Le commerce international joue un rôle clé dans le développement économique des nations. Les accords de libre-échange, les barrières tarifaires et la compétitivité des entreprises déterminent l'évolution des exportations et des importations. Une balance commerciale excédentaire peut renforcer la devise d'un pays, tandis qu'un déficit prolongé peut entraîner une dévaluation et une dépendance accrue aux capitaux étrangers.",
    
    "Les marchés financiers sont essentiels pour le financement des entreprises et des gouvernements. Les bourses permettent aux entreprises de lever des fonds en émettant des actions, tandis que les obligations offrent aux investisseurs un moyen sûr de générer des revenus. Les fluctuations des indices boursiers sont souvent influencées par les annonces économiques, les résultats d'entreprises et les tensions géopolitiques.",
    
    "L'inflation et la déflation sont des phénomènes économiques qui influencent le pouvoir d'achat des ménages. Une inflation modérée est généralement bénéfique pour l'économie, car elle encourage la consommation et l'investissement. Cependant, une inflation excessive peut éroder les salaires et provoquer une crise du coût de la vie, tandis que la déflation peut entraîner une stagnation économique en réduisant les dépenses des consommateurs et des entreprises.",
    
    "Le marché du travail est un indicateur clé de la santé économique. Le taux de chômage reflète la capacité d'une économie à créer des emplois et à absorber la main-d'œuvre disponible. Les politiques de l'emploi, la formation professionnelle et les nouvelles dynamiques du travail, comme le télétravail et l'automatisation, influencent la stabilité et l'évolution de l'emploi dans différents secteurs.",
    
    "Les politiques économiques varient selon les modèles adoptés par les pays. Certains privilégient une économie de marché avec une intervention minimale de l'État, tandis que d'autres optent pour un modèle mixte où le gouvernement joue un rôle actif dans la régulation et la redistribution des richesses. Les choix économiques ont des répercussions sur la croissance, les inégalités sociales et la stabilité financière à long terme."
]

# Fonction de prétraitement du texte
def preprocess_corpus(corpus):
    processed_sentences = []
    for sentence in corpus:
        doc = nlp(sentence)
        processed_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        processed_sentences.append(processed_tokens)
    return processed_sentences



# Prétraiter les corpus
processed_corpus_1 = preprocess_corpus(corpus_1)
processed_corpus_2 = preprocess_corpus(corpus_2)


def count(tokens):
    token_count = {}
    for token in tokens:
        token_count[token] = token_count.get(token, 0) + 1
    return token_count

def Formule_TF(tokens):
    token_count = count(tokens)
    total_tokens = len(tokens)
    return {word: count / total_tokens for word, count in token_count.items()}

def Formule_IDF(documents):
    num_documents = len(documents)
    unique_words = set(word for doc in documents for word in doc)
    return {word: math.log(num_documents / (1 + sum(1 for doc in documents if word in doc))) for word in unique_words}


# Construire la matrice TF-IDF
def compute_tfidf(corpus):
    tf_values = [Formule_TF(tokens) for tokens in corpus]
    idf_values = Formule_IDF(corpus)
    tfidf_values = [{word: tf[word] * idf_values[word] for word in tf} for tf in tf_values]
    return tfidf_values, idf_values

# Calculer les matrices TF-IDF
tfidf_corpus_1, idf_corpus_1 = compute_tfidf(processed_corpus_1)
tfidf_corpus_2, idf_corpus_2 = compute_tfidf(processed_corpus_2)

# Construire la matrice TF-IDF pour K-Means
unique_words = sorted(set(word for doc in processed_corpus_1 + processed_corpus_2 for word in doc))
def build_tfidf_matrix(tfidf_values, unique_words):
    return [[tfidf.get(word, 0) for word in unique_words] for tfidf in tfidf_values]

tfidf_matrix_1 = build_tfidf_matrix(tfidf_corpus_1, unique_words)
tfidf_matrix_2 = build_tfidf_matrix(tfidf_corpus_2, unique_words)

tfidf_matrix = np.array(tfidf_matrix_1 + tfidf_matrix_2)

# Standardisation des données
scaler = StandardScaler()
tfidf_matrix_scaled = scaler.fit_transform(tfidf_matrix)

# Application de K-Means
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(tfidf_matrix_scaled)
labels = kmeans.labels_

# Analyse des clusters
cluster_corpus_1 = labels[:len(corpus_1)]
cluster_corpus_2 = labels[len(corpus_1):]

# Vérifier si les corpus sont bien séparés
similarity = np.mean(cluster_corpus_1 == cluster_corpus_2)
if similarity > 0.5:
    conclusion = "Les corpus sont similaires."
else:
    conclusion = "Les corpus ne sont pas similaires."

logging.info(conclusion)




# Calcul de la similarité cosinus entre les matrices TF-IDF des deux corpus
similarity_matrix = cosine_similarity(tfidf_matrix_1, tfidf_matrix_2)

# Tracer la matrice de similarité
plt.figure(figsize=(10, 8))
plt.imshow(similarity_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar(label='Cosinus Similarité')
plt.title("Matrice de Cosinus Similarité entre les deux corpus")
plt.xlabel("CORPUS 2")
plt.ylabel("CORPUS 1")
plt.xticks(range(len(corpus_2)), [f"Doc {i+1}" for i in range(len(corpus_2))], rotation=90)
plt.yticks(range(len(corpus_1)), [f"Doc {i+1}" for i in range(len(corpus_1))])
plt.tight_layout()
plt.show()
