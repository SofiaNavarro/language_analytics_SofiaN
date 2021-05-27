import os
import spacy
import argparse
from pathlib import Path
nlp = spacy.load("en_core_web_sm")
import pandas as pd
import numpy
import matplotlib.pyplot as plt
from spacytextblob.spacytextblob import SpacyTextBlob

spacy_text_blob = SpacyTextBlob()
nlp.add_pipe(spacy_text_blob)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", default = Path("../language_data/A1_data/corpus"), help="Path to datafolder")
    parser.add_argument('-s', '--sample_num', type = int, help='Run the script on a smaller sample of headlines')
    args = parser.parse_args()

    # Define path to the csv file
    headlines_file = os.path.join("..", "language_data", "A3_data.csv")

    # Read the file using pandas
    mil_head = pd.read_csv(headlines_file)

    # Select only the titles column of the dataframe
    mil_titles = mil_head["headline_text"]
    
    # select only the publish dates column
    mil_dates = mil_head["publish_date"]
    
    
    # Running on just a sample of headlines
    if args.sample_num:
        mil_titles = mil_titles[:args.sample_num]
        mil_dates = mil_dates[:args.sample_num]

    # Make empty list to which the polarity scores will later be appended
    polarity = []

    # Pipe the headlines in batches of 1000 and calculate polarity score for each of them
    for doc in nlp.pipe(mil_titles, batch_size=1000):
        for sentence in doc.sents:
            score = doc._.sentiment.polarity
            polarity.append(score)


    # Zip dates and polarity together
    co_date_pol = zip(mil_dates, polarity)

    # Convert co_date_pol into a dataframe using pandas
    data_frame = pd.DataFrame(co_date_pol, columns = ("publish_date", "polarity"))

    # Because there are an uneven number of headlines per day, we collapse the polarity scores
    # of each day into one value corresponding to the mean, so we only have one value per day
    data_frame = data_frame.groupby("publish_date", as_index=False).mean()

    # The publish dates are in integer format, so we change them to an appropriate date
    # format, so Python doesn't mess up the timeline
    data_frame['publish_date'] = pd.to_datetime(data_frame['publish_date'], format='%Y%m%d')


    # Make two copies of the dataframe, one for the weekly rolling mean and one for monthly rolling mean
    weekly = pd.DataFrame.copy(data_frame)
    monthly = pd.DataFrame.copy(data_frame)

    # calulate daily rolling mean and mutate the polarity column by saying 
    # it should equal to itself plus calling the mean function
    weekly["polarity"]=weekly["polarity"].rolling(7).mean()

    # calculate monthly rolling mean and mutate the polarity column by saying 
    # it should equal to itself plus calling the mean function
    monthly["polarity"]=monthly["polarity"].rolling(30).mean()

    # Make a figure to plot the rolling means
    plt.figure(figsize=(20,10))

    # Create more values on the y-axis (for easier reading)
    plt.locator_params(axis='y', nbins=20)

    # plot weekly and monthly rolling average and provide labels for the legend
    plt.plot(weekly['publish_date'], weekly['polarity'], label = 'Weekly rolling average')
    plt.plot(monthly['publish_date'], monthly['polarity'], label = 'Monthly rolling average')

    # Give the plot a title
    plt.title('Weekly and monthly smoothing of headline polarity scores')

    # Provide names for the x and y axes
    plt.xlabel('Date')
    plt.ylabel('Polarity score')
    plt.legend()

    # Show the plot
    plt.show()

    # Save the plot
    plt.savefig(os.path.join('..', 'language_data', 'A3_output', 'Polarity_scores.png'))

    print("Done, the plot showing the polarity scores should now be saved under A3_output")

if __name__ == '__main__':
    main()