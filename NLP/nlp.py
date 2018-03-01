import nltk
from nltk.tokenize import word_tokenize, PunktSentenceTokenizer
from nltk.corpus import stopwords, state_union
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# nltk.download()

##### STOP WORDS EXAMPLE #####

example_sentence = "This is an example showing off stop word filtration."

stop_words = set(stopwords.words("english"))

words = word_tokenize(example_sentence)

filter_sentence = [w for w in words if w not in stop_words]

print(filter_sentence)

##### STEAMMER EXAMPLE #####

ps = PorterStemmer()

example_words = ["pythone", "pythoner", "pythoning", "pythoned", "pythonly"]

for w in example_words:
    print(ps.stem_word(w))

##### SENTENCE TOKENIZER EXAMPLE #####

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_some_tokenizer = PunktSentenceTokenizer(train_text)

tokenized = custom_some_tokenizer(sample_text)


def proce_content():
    try:
        for w in tokenized:
            words = nltk.word_tokenize(w)
            tagget = nltk.pos_tag(words)

            print(tagget)

    except Exception as e:
        print(str(e))


proce_content()

##### LEMMATIZER EXAMPLE #####
lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize('cats'))
print(lemmatizer.lemmatize('better', pos='a'))
