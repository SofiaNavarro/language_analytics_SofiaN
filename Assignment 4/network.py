import os
import argparse
import pandas as pd
from collections import Counter
from itertools import combinations 
from tqdm import tqdm
import spacy
nlp = spacy.load("en_core_web_sm")
import networkx as nx
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,20)
import matplotlib.pyplot as plt
from itertools import combinations
from collections import Counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True, help="Path to datafolder")
    args = vars(parser.parse_args())

    # Load the csv file
    datapath = args["path"]
    data = pd.read_csv(datapath)


    # Filter out the fake news and take only the content text of the real news.
    real_df = data[data["label"]=="REAL"]["text"]

    ## Extract entities 
    text_entities = []

    for text in tqdm(real_df):
        # create temp list
        tmp_list = []
        # create doc object
        doc = nlp(text)
        # for every named entity in the doc
        for entity in doc.ents:
            if entity.label_ == "PERSON":
                tmp_list.append(entity.text)
        # add tmp_æist to main list
        text_entities.append(tmp_list)


    ## Create an edgelist

    # empty list to which we will append the edges
    edgelist = []

    # loop over every document 
    for doc in text_entities:
        # create a list that takes all the combinations in that pair
        # so a collection of all two pairs in that list
        edges = list(combinations(doc, 2))
        # for each combination, i.e. each pair of nodes
        for edge in edges:
            # append this to final edgelist
            edgelist.append(tuple(sorted(edge)))

    # we want to loop over the edgelist and
    # create a dataset that counts every occurence of each pair
    # across the whole dataset

    counted_edges = []

    # so we are counting how often entities are mentioned in a document
    for pair, weight in Counter(edgelist).items():
        nodeA = pair[0]
        nodeB = pair[1]
        counted_edges.append((nodeA, nodeB, weight))

    # Create a dataframe
    edges_df = pd.DataFrame(counted_edges, columns=["nodeA", "nodeB", "weight"])

    # filter the dataframe to include only those instances which occur more than 500 times
    filtered_df = edges_df[edges_df["weight"]>500]

    # to create a graph we take our edgelist and make a graph
    G = nx.from_pandas_edgelist(filtered_df, "nodeA", "nodeB", ["weight"])

    # plot it
    figure = nx.draw_random(G, with_labels=True, node_size=20, font_size=10)

    # Make viz outpath
    outpath_viz = os.path.join('..', 'viz',' network.png')

    # Save plot
    nx.draw(G, with_labels=True, node_size=20, font_size=10)
    plt.savefig(outpath_viz, dpi=300, bbox_inches="tight")

    ## Calculate centrality measures

    bc_metric = nx.betweenness_centrality(G)
    ev_metric = nx.eigenvector_centrality(G)

    # Make dataframe
    bc_dataframe = pd.DataFrame(bc_metric.items(), columns=["node", "betweenness"]).sort_values("betweenness", ascending=False)

    ev_dataframe = pd.DataFrame(ev_metric.items(), columns=["node", "eigenvector"]).sort_values("eigenvector", ascending=False)


    # Make output for centrality measures
    outpath_cen = os.path.join('..', 'measures', 'measures.csv')

    # Save the centrality measure dataframes to the output
    bc_dataframe.to_csv("outpath_cen")
    ev_dataframe.to_csv("outpath_cen")

    print("Done, the weighted edge-plot should be saved under the 'viz' folder, and the .csv file containing the centrality measures are in the outpath_cen folder")

if __name__ == '__main__':
    main()
