"""
Python script for post tagging learning
"""
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

train_path = "dataset/UD_Indonesian/id-ud-train.conllu"
test_path = "dataset/UD_Indonesian/id-ud-dev.conllu"

def features(sentence, index):
    """ sentence: [w1, w2, ...], index: the index of the word """
    return {
        'word': sentence[index],
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'is_capitalized': sentence[index][0].upper() == sentence[index][0],
        'is_all_caps': sentence[index].upper() == sentence[index],
        'is_all_lower': sentence[index].lower() == sentence[index],
        'prefix-1': sentence[index][0],
        'prefix-2': sentence[index][:2],
        'prefix-3': sentence[index][:3],
        'suffix-1': sentence[index][-1],
        'suffix-2': sentence[index][-2:],
        'suffix-3': sentence[index][-3:],
        'prev_word': '<s>' if index == 0 else sentence[index - 1],
        'next_word': '</s>' if index == len(sentence) - 1 else sentence[index + 1],
        'has_hyphen': '-' in sentence[index],
        'is_numeric': sentence[index].isdigit(),
        'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
    }

def untag(tagged_sentence):
    return [w for w, t in tagged_sentence]

def gen_corpus(path):
    doc = []
    with open(path) as file:
        for line in file:
            if line[0].isdigit():
                features = line.split()
                word, pos = features[1], features[3]
                if pos != "_":
                    doc.append((word, pos))
            elif len(line.strip()) == 0:
                if len(doc) > 0:
                    words, tags = zip(*doc)
                    yield (list(words), list(tags))
                doc = []

def evaluation(TEST_DATA):
    y_pred, y_true = [], []
    for words, tags in TEST_DATA:
        for i, (word, pos) in enumerate(pos_tag(words)):
            y_pred.append(pos)
            y_true.append(tags[i])
    return y_pred, y_true

def transform_to_dataset(tagged_sentences):
    X, y = [], []

    for words, tags in tagged_sentences:
        for index, word in enumerate(words):
            X.append(features(words, index))
            y.append(tags[index])
    return X, y

def pos_tag(sentence):
    tags = clf.predict([features(sentence, index) for index in range(len(sentence))])
    return zip(sentence, tags)

#Using Validation dataset
# tagged_sentences = list(gen_corpus(train_path))
# cutoff = int(.75 * len(tagged_sentences))
# training_sentences = tagged_sentences[:cutoff]
# test_sentences = tagged_sentences[cutoff:]

#Using Test data
training_sentences = list(gen_corpus(train_path))
test_sentences = list(gen_corpus(test_path))

print (len(training_sentences))   # 14554
print (len(test_sentences))    # 298

X, y = transform_to_dataset(training_sentences)
print(len(X)) #  nb_features : 356419

clf = Pipeline([
    ('vectorizer', DictVectorizer(sparse=True)),
    ('classifier',  LogisticRegression(n_jobs=4, max_iter=200, verbose=True))
])

clf.fit(X, y)

X_test, y_test = transform_to_dataset(test_sentences)
print( "Accuracy:", clf.score(X_test, y_test)) # Accuracy: 0.951851851852

# test
sents = ["Setelah mengamankan YA dan staf serta uang Rp 800 juta, KPK mendatangi kantor Cilegon United Football Club dan mengamankan uang Rp 352 juta",
         "Warga Desa Kebondalem, Kota Cilegon, Aryo Wibisono, juga menyebutkan Cilegon butuh penanganan khusus dari KPK",
         "Rossi mampu menjadi pembalap tercepat ketiga di kualifikasi MotoGP Aragon",
         "Seperti dilansir Calciomercato, Belotti telah berbicara dengan manajemen Il Toro. Pemain berusia 23 tahun itu mempertimbangkan untuk hijrah ke klub London Barat tersebut"]

for s in sents:
    for w, pos in pos_tag(s.split()):
        print("%s/%s" % (w, pos), end=' ')

y_true, y_pred = evaluation(test_sentences)
for l in classification_report(y_true, y_pred).split('\n'):
    print(l)
