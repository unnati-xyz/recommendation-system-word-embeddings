import numpy as np
import re
import pandas as pd
import numpy as np
from scipy import linalg
import pickle
import os.path
from sherlock.analytics.word_embeddings import WordEmbedding
from sherlock import model
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')

class Predictions:


    def __init__(self):
        self.word_vocab = set(model.index2word)
        self.product_vector_df = self.get_product_vector_df()


    def get_product_vector_df(self):
        if os.path.isfile('product_vector_df'):
            with open('product_vector_df', 'rb') as inpFile:
                df = pickle.load(inpFile)
                return df
        else:
            we = WordEmbedding()
            we.generate_artist_tag_tokens()
            product_data = []
            for product in we.cleaned_tags:
                feature_vector = np.zeros((100,), dtype="float32")
                nwords = len(product)
                for token in product:
                    feature_vector = np.add(feature_vector, model[token])

                feature_vector = np.divide(feature_vector, nwords)
                if len(product) == 0:
                    print(product)
                product_data.append({'tags': product, 'vector': feature_vector})

            df = pd.DataFrame(product_data, columns=['tags', 'vector'])
            print("saving file")
            with open('product_vector_df' , 'wb') as fp:
                pickle.dump(df, fp)

            return df


    def get_tag_prediction(self, tags, topn=5):
        words_present_in_index = []
        for tag in tags:
            if tag.lower() in self.word_vocab:
                words_present_in_index.append(tag.lower())
        out = []
        for tup in model.most_similar(positive=words_present_in_index, topn=topn):
            out.append(tup[0])

        print(out)
        return out

    def get_search_query_product(self, search_query, topn=5):
        try:
            query = tokenizer.tokenize(search_query)
            search_vector = np.zeros((100,), dtype="float32")
            for tag in query:
                if len(tag) > 1 and tag not in WordEmbedding().stop_words:
                    if tag in self.word_vocab:
                        search_vector = np.add(search_vector, model[tag])

            self.product_vector_df['cosine_similarity'] = self.product_vector_df.apply(get_cosine_similarity, axis=1,
                                                                          args=(search_vector,))

            df = self.product_vector_df.sort(columns='cosine_similarity', ascending=False)[:topn]

            return df[['tags','cosine_similarity']].to_json(orient="records")

        except Exception as e:
            print(e)

def get_cosine_similarity(row, search_vector):
    try:
        cosine_similarity = np.dot(row['vector'], search_vector)/linalg.norm(row['vector'])/linalg.norm(search_vector)
        return cosine_similarity
    except Exception as e:
        raise e

if __name__ == '__main__':
    prediction = Predictions()
    #print(prediction.product_vector_df.head())
    #prediction.get_tag_prediction(tags=['pokemon', 'pikachu'], topn=5)
    a = prediction.get_search_query_product(search_query='hot dog cat man', topn=10)
    print(a)









