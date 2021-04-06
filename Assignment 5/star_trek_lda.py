# standard library
import sys, os
sys.path.append(os.path.join(".."))
from pprint import pprint
import json

# data and nlp
import pandas as pd
import spacy
nlp = spacy.load("en_core_web_sm")

# visualisation
import seaborn as sns
from matplotlib import rcParams
# figure size in inches
rcParams['figure.figsize'] = 20,10


# LDA tools
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from utils import lda_utils

# warnings
import logging, warnings
warnings.filterwarnings('ignore')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)


def main():
    # Load in the dataset containing all the lines from all the characters of all the Star Trek series
    # Unfortunately, some newlines have been parsed wrong, so that some words are 'glued' togehter. E.g. "camefrom"
    with open('all_series_lines.json') as file:
        content = file.read()
        line_dict = json.loads(content)

    # make an empty dictionary
    episodes = {}

    # loop through all the series
    for series_name, series in line_dict.items():
    # loop through all episodes (both the name, and the content)
        for episode_name, episode in series.items():
    # make empty string so we can later add all the characters lines to
            episode_string = ''
            for character_lines in episode.values():
                    lines = ' '.join(character_lines)
                # only add the lines if the character has a line, omit all empty lines.
                    if len(lines) !=0:
                        episode_string += ' '+lines
            episode_key = series_name+'_'+episode_name.split()[1]
            episodes[episode_key] = episode_string

    episodes_content = list(episodes.values())
    
    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(episodes_content, min_count=10, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[episodes_content], threshold=100)  
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    
    #process the lines to only include nouns
    processed_lines = lda_utils.process_words(episodes_content,nlp, bigram_mod, trigram_mod, allowed_postags=["NOUN"])
    
    # Convert every token to a numerical ID
    id2word = corpora.Dictionary(processed_lines)

    # Create Corpus: Term Document Frequency
    # match each ID to words and count the frequency of the tokens
    corpus = [id2word.doc2bow(episode) for episode in processed_lines]
    
    # Build LDA model
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=12, 
                                           random_state=100,
                                           chunksize=10,
                                           passes=10,
                                           iterations=100,
                                           per_word_topics=True, 
                                           minimum_probability=0.0)
    
    # Compute Perplexity
    print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, 
                                         texts=processed_lines, 
                                         dictionary=id2word, 
                                         coherence='c_v')

    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)
    
    # Can take a long time to run.
    model_list, coherence_values = lda_utils.compute_coherence_values(texts=processed_lines,
                                                                      corpus=corpus, 
                                                                      dictionary=id2word,  
                                                                      start=5, 
                                                                      limit=30,  
                                                                      step=2)
    
    df_topic_keywords = lda_utils.format_topics_sentences(ldamodel=lda_model, 
                                                      corpus=corpus, 
                                                      texts=processed_lines)

    # Format
    df_dominant_topic = df_topic_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
    df_dominant_topic.sample(10)
    
    values = list(lda_model.get_document_topics(corpus))
    
    split = []
    for entry in values:
        topic_prevelance = []
        for topic in entry:
            topic_prevelance.append(topic[1])
        split.append(topic_prevelance)
            
    df = pd.DataFrame(map(list,zip(*split)))
    
    # visualize rolling mean of the topics
    topic_plot = sns.lineplot(data=df.T.rolling(20).mean())
    topic_plot.figure.savefig('topics.png')
    
if __name__ == '__main__':
    main()
