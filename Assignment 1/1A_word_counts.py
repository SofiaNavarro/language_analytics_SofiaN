import os
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", default = Path("../language_data/A1_data/corpus"), help="Path to datafolder")
    args = parser.parse_args()

    directory = args.path # point to location of novels
    novels_names = os.listdir(directory) # Make a list of the file names
    novels_names.sort() # Sort novel titles (they weren't sorted by default)
    outpath = os.path.join("..", "language_data", "A1_output", "word_count.csv")
    

    titles = " ".join(["filename", "unique_words", "total_words"]) # creates titles for the list

    with open(outpath, "w", encoding="utf-8") as file:  # save titles in the outpath
        file.write(titles + "\n")

    for novel in novels_names:   # A forloop that goes through every novel in the list defined above
        novel_path = os.path.join(directory, novel) # Point to directory and create variable 'novel'
        with open(novel_path, "r", encoding="utf-8") as file: # Open novel folder
            each_novel = file.read() # Each novel in folder is 'read'
            split_novel = each_novel.split() # Split content of each novel on whitespace
            unique_words = set(split_novel) # Convert all elements in split_novel to a set, so each element only occurs once
            amount_words = [novel, str(len(unique_words)), str(len(split_novel))] # Print total amount of unique words & total amount of words
            save_file = " ".join(amount_words) # append novel title + total words + unique words in a string
            with open(outpath, "a", encoding="utf-8") as file: # save the whole list in the outpath
                file.write(save_file + "\n")
                
    print("Done, the csv file containing the word counts can be found in the A1_output folder")
                  

if __name__ == '__main__':
    main()
