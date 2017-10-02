"""
Python script for post tagging learning
"""
from hashlib import md5
from os.path import isfile
from time import perf_counter

from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.externals import joblib

from nltk.tag.hmm import HiddenMarkovModelTagger
from nltk.tag.perceptron import PerceptronTagger

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
    with open(path, encoding='utf-8') as file:
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

def evaluation(clf, TEST_DATA):
    y_pred, y_true = [], []
    for words, tags in TEST_DATA:
        for i, (word, pos) in enumerate(pos_tag(clf, words)):
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

def pos_tag(clf, sentence):
    tags = clf.predict([features(sentence, index) for index in range(len(sentence))])
    return zip(sentence, tags)

def logreg(train_path, test_path, sents):
    modelref = 'logreg-' + md5(('logreg///' + train_path).encode()).hexdigest() + '.pickle'

    test_sentences = list(gen_corpus(test_path))

    if not isfile(modelref):
        start = perf_counter()
        training_sentences = list(gen_corpus(train_path))
        X, y = transform_to_dataset(training_sentences)

        clf = Pipeline([
            ('vectorizer', DictVectorizer(sparse=True)),
            ('classifier',  LogisticRegression())
        ])

        clf.fit(X, y)

        end = perf_counter()
        print('Training took {} ms.'.format(int((end - start) * 1000)))
        with open(modelref, 'wb') as wf:
            joblib.dump(clf, wf)
    else:
        with open(modelref, 'rb') as rf:
            clf = joblib.load(rf)
        print('Model loaded from file.')

    for s in sents:
        for w, pos in pos_tag(clf, s.split()):
            print("%s/%s" % (w, pos), end=' ')

    if sents: print()

    start = perf_counter()
    y_pred, y_true = evaluation(clf, test_sentences)
    end = perf_counter()
    print('Testing took {} ms.'.format(int((end - start) * 1000)))

    for l in classification_report(y_true, y_pred).split('\n'):
        print(l)

def convert_sents_to_zipped(unzipped_sents):
    for words, tags in unzipped_sents:
        yield list(zip(words, tags))

def hmm(train_path, test_path):
    training_sentences = list(gen_corpus(train_path))
    test_sentences = list(gen_corpus(test_path))

    start = perf_counter()

    hmm_model = HiddenMarkovModelTagger.train(list(convert_sents_to_zipped(training_sentences)))

    end = perf_counter()
    print('Training took {} ms.'.format(int((end - start) * 1000)))
    start = perf_counter()
    # Evaluation
    y_pred, y_true = [], []
    for words, tags in test_sentences:
        y_pred.extend(y for x, y in hmm_model.tag(words))
        y_true.extend(tags)

    end = perf_counter()
    print('Testing took {} ms.'.format(int((end - start) * 1000)))

    for l in classification_report(y_true, y_pred).split('\n'):
        print(l)

def ap(train_path, test_path):
    modelref = 'ap-' + md5(('ap///' + train_path).encode()).hexdigest() + '.pickle'

    test_sentences = list(gen_corpus(test_path))

    if not isfile(modelref):
        start = perf_counter()
        training_sentences = list(gen_corpus(train_path))

        ap_model = PerceptronTagger(load=False)
        ap_model.train(list(convert_sents_to_zipped(training_sentences)), save_loc=modelref)
        end = perf_counter()
        print('Training took {} ms.'.format(int((end - start) * 1000)))
    else:
        ap_model = PerceptronTagger(load=False)
        ap_model.load(modelref)
        print('Model loaded from file.')

    # Evaluation
    start = perf_counter()
    y_pred, y_true = [], []
    for words, tags in test_sentences:
        y_pred.extend(y for x, y in ap_model.tag(words))
        y_true.extend(tags)

    end = perf_counter()
    print('Testing took {} ms.'.format(int((end - start) * 1000)))

    for l in classification_report(y_true, y_pred).split('\n'):
        print(l)

if __name__ == '__main__':
    sents = ["Setelah mengamankan YA dan staf serta uang Rp 800 juta, KPK mendatangi kantor Cilegon United Football Club dan mengamankan uang Rp 352 juta",
             "Warga Desa Kebondalem, Kota Cilegon, Aryo Wibisono, juga menyebutkan Cilegon butuh penanganan khusus dari KPK",
             "Rossi mampu menjadi pembalap tercepat ketiga di kualifikasi MotoGP Aragon",
             "Seperti dilansir Calciomercato, Belotti telah berbicara dengan manajemen Il Toro. Pemain berusia 23 tahun itu mempertimbangkan untuk hijrah ke klub London Barat tersebut"]

    print('# Logistic Regression')
    print('## Indonesian')
    logreg("dataset/UD_Indonesian/id-ud-train.conllu", "dataset/UD_Indonesian/id-ud-dev.conllu", [])
    print('## English')
    logreg("dataset/UD_English/en-ud-train.conllu", "dataset/UD_English/en-ud-dev.conllu", [])

    print('# Hidden Markov Model')
    print('## Indonesian')
    hmm("dataset/UD_Indonesian/id-ud-train.conllu", "dataset/UD_Indonesian/id-ud-dev.conllu")
    print('## English')
    hmm("dataset/UD_English/en-ud-train.conllu", "dataset/UD_English/en-ud-dev.conllu")

    print('# Averaged Perceptron')
    print('## Indonesian')
    ap("dataset/UD_Indonesian/id-ud-train.conllu", "dataset/UD_Indonesian/id-ud-dev.conllu")
    print('## English')
    ap("dataset/UD_English/en-ud-train.conllu", "dataset/UD_English/en-ud-dev.conllu")
