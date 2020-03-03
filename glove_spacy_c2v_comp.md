# Word embeddings with code2vec, glove and spacy.

One of the powerful ways to improve your machine learning model is to use word embeddings. It is able to capture the context of the word in the document and find semantic and syntactic similarity. There are many possible technics to learn word embeddings. So, one can find suitable technics for your own case. In this post, I want to cover an unusual application of the word embeddings technics. I try to find the best word embedding technics for open API specifications. In this example
I will use a free source of open API specification [apis-guru](https://apis.guru/). 

The biggest challenge is that open API specification is neither a natural language nor a code. So, we are free 
to use any of the available 
embeddings models. For my experiment, I choose three possible candidates that may work in this case: code2vec, glove, and spacy. 
[Code2vec](https://urialon.cswp.cs.technion.ac.il/wp-content/uploads/sites/83/2018/12/code2vec-popl19.pdf) is a neural model that 
learn analogies that are relevant to source code. It was  trained on the java code database but can be applied to any code. [Glove](https://nlp.stanford.edu/projects/glove/) is 
a commonly used algorithm for the NLP task trained on Wikipedia and [Gigawords](https://github.com/harvardnlp/sent-summary).
And last but not least is [Spacy](https://spacy.io/usage/vectors-similarity).  It is a recently developed algorithm, but it already has a reputation of the fastest word embedding in the word. Leet's see which of these algorithms is better for Open API datasets and which one works faster for open API specification.

I divided this post into six sections, each of them will contain code examples and some tips for future use:

1. Upload dataset.
2. Upload vocabularies.
3. Extract the field's name.
4. Remove duplicates from the list of field names.
5. Tokenize keys, by removing - and_, and by splitting camel case words.
6. Create a dataset of the names of the fields.
7. Check different embeddings models.
8. Similarity check.
9. Conclusion.


Now, we can start.

## 1. Upload dataset. 


First let's make some imports.

```
import string
from gensim.models import KeyedVectors as word2vec
import numpy as np
from openapi_typed import OpenAPIObject, Response, Reference, Parameter, RequestBody, Header, Components, Schema, Paths
from typing import Sequence, Mapping, Union, cast
import yaml
import os
import posixpath
```
Important! You need ```posixpath ``` only if you want to manipulate with Windows path on Unix machine or vice versa.  

We are all set. Most of the apis-guru specifications are in Swagger 2.0 format. But.. the latest version of OpenAPI specification is OpenAPI 3.0. Lets convert the whole dataset to this format by using [Unmock scripts](https://github.com/meeshkan/unmock-openapi-scripts)! This may take a while but in the end, you will get a big dataset with various specifications. 

## 2. Upload vocabularies.
### Code2Vec
1. Download code2vec model.
2. Load by using gensim library.
```model = word2vec.load_word2vec_format(vectors_text_path, binary=False)```
### Glove
1. Download one of the Glove vocabularies from the website.
2. Load Glove vocabulary manually.
```
embeddings_dict = {}
with open("../glove/glove.6B.300d.txt", 'r', encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector
```
### Spacy
Load large Spacy vocabulary
```nlp = spacy.load('en_core_web_lg')```.

## 3. Extract field's names. 

The whole list of files can be obtained from ```scripts/fetch-list.sh``` file or by using the following function(for Windows):
```
def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = posixpath.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if posixpath.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles
```
Another big deal is to get field names out of open API specifications. For this purpose, we will use [openapi-typed library](https://pypi.org/project/openapi-typed-2/).  Let's define a function ```get_fields ``` that takes the open API specification and returns a list of field names. 
```
def get_fields_from_schema(o: Schema) -> Sequence[str]:
    return [
        *(o['properties'].keys() if ('properties' in o) and (type(o['properties']) == type({})) else []),
        *(sum([
            get_fields_from_schema(schema) for schema in o['properties'].values() if not ('$ref' in schema) and type(schema) == type({})], []) if ('properties' in o) and ($        *(get_fields_from_schema(o['additionalProperties']) if ('additionalProperties' in o) and (type(o['additionalProperties']) == type({})) else []),
        *(get_fields_from_schema(o['items']) if ('items' in o) and  (type(o['items'] == type({}))) else []),
    ]


def get_fields_from_schemas(o: Mapping[str, Union[Schema, Reference]]) -> Sequence[str]:
    return sum([get_fields_from_schema(cast(Schema, maybe_schema)) for maybe_schema in o.values() if not ('$ref' in maybe_schema) and (type(maybe_schema) == type({}))], [])
def get_fields_from_responses(o: Mapping[str, Union[Response, Reference]]) -> Sequence[str]:
    return []

def get_fields_from_parameters(o: Mapping[str, Union[Parameter, Reference]]) -> Sequence[str]:
    return []

def get_fields_from_request_bodies(o: Mapping[str, Union[RequestBody, Reference]]) -> Sequence[str]:
    return []

def get_fields_from_headers(o: Mapping[str, Union[Header, Reference]]) -> Sequence[str]:
    return []

def get_fields_from_components(o: Components) -> Sequence[str]:
    return [
        *(get_fields_from_schemas(o['schemas']) if 'schemas' in o else []),
        *(get_fields_from_responses(o['responses']) if 'responses' in o else []),
        *(get_fields_from_parameters(o['parameters']) if 'parameters' in o else []),
        *(get_fields_from_request_bodies(o['requestBodies']) if 'requestBodies' in o else []),
        *(get_fields_from_headers(o['headers']) if 'headers' in o else []),
            ]                                                                                                                                                                       
def get_fields_from_paths(o: Paths) -> Sequence[str]:
    return []

def get_fields(o: OpenAPIObject) -> Sequence[str]:
    return [
        *(get_fields_from_components(o['components']) if 'components' in o else []),
        *(get_fields_from_paths(o['paths']) if 'paths' in o else []),
    ] 
```
## 4. Remove duplicates from the list. 
Note, that some of the field names may repeat even in one specification.  In order to get rid of repetition let's convert the list of field names from the list to the dictionary and back by using ```list(dict.fromkeys(col))```.

##  5. Tokenization
Some of the field names may contain punctuation, such as _ and - symbols, or camel case words. These words can be chopped up into pieces called tokens.  The function of ```camel-case``` identifies camel cases. First, it checked is there any punctuation, if yes, then it is not a camel case. Then we should check is there capital letters inside the word (first and last characters are excluded). 

```
def camel_case(example):      
    if  any(x in example for x  in string.punctuation)==True:
        return False
    else:
        if any(list(map(str.isupper, example[1:-1])))==True:
            return True
        else:
            return False
```
The following function split the camel case into pieces. For this purpose, we should identify the upper case and mark places where the case changes. The function returns a list of the words after splitting. For example field name "BodyAsJson" transforms to a list ['Body', 'As', 'Json']. 
```
def camel_case_split(word):
    idx = list(map(str.isupper, word))
    case_change = [0]
    for (i, (x, y)) in enumerate(zip(idx, idx[1:])):
        if x and not y:  
            case_change.append(i)
        elif not x and y:  
            case_change.append(i+1)
    case_change.append(len(word))
    return [word[x:y] for x, y in zip(case_change, case_change[1:]) if x < y]
```
The camel_case_split function is used in the following tokenization algorithm. Here we check if punctuation is in the word, then we split the word into pieces. Each piece can be a camel case word. If it is the case we split it into smaller pieces. After splitting each element of the list converted to the lower case for convenience. 

```
def tokenizer(mylist):
    tokenized_list=[]
    for word in mylist:

        if '_'  in word:
            splitted_word=word.split('_')
            for elem in splitted_word:
                if camel_case(elem)==True:
                    elem=camel_case_split(elem)
                    for el1 in elem:
                        tokenized_list.append(el1.lower())
                else:    
                    tokenized_list.append(elem.lower())
        elif '-' in word:
            hyp_word=word.split('-')
            for i in hyp_word:
                if camel_case(i)==True:
                    i=camel_case_split(i)
                    for el2 in i:
                        tokenized_list.append(el2.lower())
                else: 
                    tokenized_list.append(i.lower())
        elif camel_case(word)==True:
            word=camel_case_split(word)
            for el in word:
                tokenized_list.append(el.lower())
        else:
            tokenized_list.append(word.lower())
    return(tokenized_list)
```

## 6. Create a dataset of the names of the fields.

Now, let's create a big dataset with fields name from all the specifications. The  ```dict_dataset``` function takes a list of the file's name and path and opens each specification file.  For each file the get_field function returns a list of the names of the fields, then we remove duplicates from the list, and tokenize it. In the end, we create a dictionary with a file name as a key and list of fields name as a value. 

```
def dict_dataset(datasets):
    dataset_dict={}
    for i in datasets:
        with open(i, 'r') as foo:
            col=algo.get_fields(yaml.safe_load(foo.read()))
            if col:
                mylist = list(dict.fromkeys(col))
                tokenized_list=tokenizer(mylist)
                dataset_dict.update({i: tokenized_list})
            else:
                continue
    return (dataset_dict)
```

## 7. Embeddings test.



Now we can find out-of-vocabulary words(not_identified_c2v) and count the percentage of these words for code2vec vocabulary.
```
not_identified_c2v=[]
count_not_indent=[]
total_number=[]

for ds in test1:
    count=0
    for i in data[ds]:
        try:
            model[i]
        except KeyError:
            not_identified_c2v.append(i)
            count+=1
    count_not_indent.append(count)
    total_number.append(len(data[ds]))

total_code2vec=sum(count_not_indent)/sum(total_number)*100
```
The same code will word for Glove. Spacy vocabulary is different, so we need to modify our code accordingly.
```
not_identified_sp=[]
count_not_indent=[]
total_number=[]

for ds in test1:
    count=0
    for i in data[ds]:
        if (i in nlp.vocab)== False:
                count+=1
                not_identified_sp.append(i)
    count_not_indent.append(count)
    total_number.append(len(data[ds]))

#    print(ds, count, len(data[ds]))
        
total_spacy=sum(count_not_indent)/sum(total_number)*100
```
The resulting percentages of not identified words are ``` 3.39, 2.33, 2.09% ``` for code2vec, glove, and spacy, respectively. Since the percentages are relatively small and similar for all algorithms, we can make another test.

First, let's create a test dictionary with the words that should be similar across all API specifications:
``` test_dictionary={'host': 'server',
'pragma': 'cache',
'id': 'uuid',
'user': 'client',
'limit': 'control',
'balance': 'amount',
'published': 'date',
'limit': 'dailylimit',
'ratelimit': 'rate',
'start': 'display',
'data': 'categories'}  
```
For Glove and Code2Vec we can use ```similar_by_vector``` method provided in the gensim library. But, Spacy does not implement it yet. But, we can find 100 most similar words on our own. Format the input vector for use in the distance function. In this case, we will create each key in the dictionary and check whether the corresponding value is in the 100 most similar words or not. First, we format the vocabulary for use in a ```distance.cdist ``` function. It is the function that computes the distance between each pair of vectors in the vocabulary. Then we sort the list from the smallest distance to largest and took the first 100 words.

```
from scipy.spatial import distance
input_word = "frog"
p = np.array([nlp.vocab[input_word].vector])


ids = [x for x in nlp.vocab.vectors.keys()]
vectors = [nlp.vocab.vectors[x] for x in ids]
vectors = np.array(vectors)

# *** Find the closest word below ***
    closest_index = distance.cdist(p, vectors)[0].argsort()[::-1][-100:]
    word_id = [ids[closest_ind] for closest_ind in closest_index]
    output_word = [nlp.vocab[i].text for i in word_id]
    #output_word
    list1=[j.lower() for j in output_word]
    mylist = list(dict.fromkeys(list1))
    count=0
    if test_dictionary[k] in mylist:
        count+=1
        print(k,count, 'yes')
    else:
        print(k, 'no')
```
## Conclusion

We have tried three word embeddings algorithms for Open API specification. Despite the fact, that all three perform quite well on this dataset, an extra comparison of the most similar words shows that Spacy works better for our case. 

Spacy is faster than other algorithms. Spacy vocabulary can be upload five times faster in comparison to Glove or Code2Vec vocabularies. But, the lack of functions, such as similar_by_vector and similar_word  make obstacles in using this algorithm.

In conclusion, I should say that the fact that Spacy works with this dataset does not mean that Spacy will be better for all-existed-in-the-word datasets. So,  feel free to try different word embeddings for your own dataset and let me know which one works better for you in the comments!

## Thanks for reading! 





