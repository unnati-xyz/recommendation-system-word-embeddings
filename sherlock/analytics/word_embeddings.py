from gensim.models import Word2Vec
import pickle
from nltk.tokenize import RegexpTokenizer
import traceback
tokenizer = RegexpTokenizer(r'\w+')


class WordEmbedding:

    def __init__(self):
        with open('stopwords', 'rb') as fp:
            self.stop_words = pickle.load(fp)
            self.cleaned_tags = []



    def generate_artist_tag_tokens(self):
        with open('data/artist_tags.csv', 'r') as inpFile:
            lines = inpFile.readlines()
            for line in lines:
                if line.isspace() is False:
                    tags = tokenizer.tokenize(line)
                    final_tags = []
                    if len(tags) > 0:
                        for tag in tags:
                            if len(tag) > 1 and tag not in self.stop_words:
                                final_tags.append(tag.lower())

                        if len(final_tags) > 0:
                            self.cleaned_tags.append(list(set(final_tags)))


    def train_model(self):
        try:
            print("Generating tags by reading artist tag files")
            self.generate_artist_tag_tokens()
            print(type(self.cleaned_tags))
            print(len(self.cleaned_tags))
            print("Generating neural network model to get word embeddings")
            word2vec_sg = Word2Vec(sentences=self.cleaned_tags, size=100, window=5, min_count=1, workers=4, sg=1, iter=10)
            word2vec_sg.save("model/word2vec_model")
        except Exception as e:
            print(traceback.format_exc())
            raise e

    def get_model(self):
        try:
                return Word2Vec.load("model/word2vec_model")

        except Exception as e:
            print(traceback.format_exc())
            raise e


if __name__ == "__main__" :
    w2v = WordEmbedding()
    w2v.train_model()
