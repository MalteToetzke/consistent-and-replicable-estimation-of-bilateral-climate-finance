import nltk
from nltk.stem import WordNetLemmatizer
import re

class Preprocess_text:
    def __init__(self):
        self.stopwords = ["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all",
                          "almost",
                          "alone", "along", "already", "also", "although", "always", "am", "among", "amongst",
                          "amoungst",
                          "amount", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere",
                          "are",
                          "around", "as", "at", "back", "be", "became", "because", "become", "becomes", "becoming",
                          "been",
                          "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond",
                          "bill",
                          "both", "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con", "could",
                          "couldnt", "cry",
                          "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight",
                          "either",
                          "eleven", "else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
                          "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire",
                          "first",
                          "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full",
                          "further",
                          "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter",
                          "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however",
                          "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself",
                          "keep",
                          "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me",
                          "meanwhile",
                          "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my",
                          "myself",
                          "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none",
                          "noone",
                          "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only",
                          "onto",
                          "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own",
                          "part", "per",
                          "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming",
                          "seems",
                          "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty",
                          "so",
                          "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still",
                          "such",
                          "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then",
                          "thence",
                          "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they",
                          "thick",
                          "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus",
                          "to",
                          "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under",
                          "until", "up",
                          "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when",
                          "whence",
                          "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever",
                          "whether",
                          "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will",
                          "with",
                          "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"]


    def preprocess_text(self,df,column,remove_stopwords=False):
        # df --> pandas array
        # remove_stopwords = False to include stopwords in preprocessed text


        print("starting")


        # Lemmatization & Noise removal in one step
        nltk.download('wordnet')

        df['text_preprocessed'] = df[column]

        # Lowercasing
        df['text_preprocessed'] = df['text_preprocessed'].str.replace(',', '')
        df['text_preprocessed'] = df['text_preprocessed'].str.replace(r"[\"\',]", '')
        # Stop Word Removal
        df['text_preprocessed'] = df['text_preprocessed'].str.lower().str.split()

        if remove_stopwords:
            df['text_preprocessed'] = df['text_preprocessed'].apply(lambda x: [item for item in x if item not in self.stopwords])

        # Function
        for i, row in df.iterrows():
            print(i)
            try:
                # remove Noise
                clean_text = [self.scrub_words(w) for w in row['text_preprocessed']]
            except:
                clean_text = "None"
                print("Presprocess failed")


            df.at[i, 'text_preprocessed'] = clean_text

        # Join back for Text
        df['text_preprocessed'] = df['text_preprocessed'].str.join(' ')
        df['text_preprocessed'] = df['text_preprocessed'].apply(self.remove_whitespace)

        return df



    def scrub_words(self, text):
        """Basic cleaning of texts."""

        # remove html markup
        text = re.sub("(<.*?>)", "", text)

        # remove other weird letters
        text = re.sub(r"\w*[□,©,$]\w*", "", text)

        # remove non-ascii and digits
        text = re.sub("(\\W|\\d)", "", text)

        # remove whitespace
        text = text.strip()
        return text


    def remove_whitespace(self, x):
        """
        Helper function to remove any blank space from a string
        x: a string
        """
        try:
            # Remove spaces inside of the string
            x = " ".join(x.split())

        except:
            pass
        return x
