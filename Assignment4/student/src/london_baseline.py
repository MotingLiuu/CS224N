# TODO: [part d]
# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.

import argparse
import utils

def main():
    accuracy = 0.0
    # Compute accuracy in the range [0.0, 100.0]
    ### YOUR CODE HERE ###
    count_correct = 0
    count_sample= 0
    with open('birth_dev.tsv', 'r', encoding='utf-8') as file:
        for line in file:
            question, place = line.strip().split('\t')
            if place == 'London':
                count_correct += 1
            count_sample += 1
    accuracy = count_correct / count_sample
    ### END YOUR CODE ###

    return accuracy

if __name__ == '__main__':
    accuracy = main()
    with open("london_baseline_accuracy.txt", "w", encoding="utf-8") as f:
        f.write(f"{accuracy}\n")
