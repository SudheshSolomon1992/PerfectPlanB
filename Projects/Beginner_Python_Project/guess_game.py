import random

alphabetList = list()
count = 0

fname = 'fruits.txt'
fhand = open(fname).read().splitlines()
text = random.choice(fhand)
text = text.lower()

def get_input():
    # get the word for guessing
    print ("\n")
    user_input = input("Enter an alphabet: ").lower()
    single_flag = checkSingleLetter(user_input)
    alpha_flag = checkAlphabet(user_input)
    if single_flag and alpha_flag:
        alphabetPresent(user_input)
    else:
        print ("Invalid input!! Accepted inputs are from a to z ")
        print ("Try again...")
        get_input()

# use text file for input

# use a function to check 
# if the user has inputted a single letter
def checkSingleLetter(user_input):
    if len(user_input) == 1:
        return True
    else:
        return False

def checkAlphabet(user_input):
    if user_input.isalpha():
        return True
    else:
        return False


# if the inputted letter is in the hidden word (if so how many times)
def alphabetPresent(alphabet):
    global count
    if alphabet in text:
        if text.count(alphabet) == 1:
            print ("Alphabet", alphabet, "is present", text.count(alphabet), "time")
        else:
            print ("Alphabet", alphabet, "is present", text.count(alphabet), "times")
        alphabetList.append(alphabet)
        printAlphabets(alphabet, alphabetList)
    else:
        print ("Alphabet", alphabet, "is not present", end=' ')
        count = count + 1

# to print letters
def printAlphabets(alphabet, alphabetList):
    indices = dict()
    
    print ("Previous alphabets: ", str(alphabetList).replace('[', '').replace(']', '').replace("'", '').replace(',', ''))

    for alphabet in alphabetList:
        str_idx = charposition(text, alphabet)
        for i in range(len(text)):
            if i in str_idx:
                indices[i] = alphabet

    print ("Word: ", end=' ')
    for i in range(len(text)):
        if i in indices.keys():
            print (text[i], end=' ')
        else:
            print ('_', end=' ')

    curr_word = ''
    for position, letter in sorted(indices.items()):
        curr_word = curr_word + letter
        
    if curr_word == text:
        print ("\nYay!!! You have won the game by finding the word correctly")
        quit()

def charposition(string, char):
    pos = [] #list to store positions for each 'char' in 'string'
    for n in range(len(string)):
        if string[n] == char:
            pos.append(n)
    return pos

# a counter variable to limit guesses
def numGuesses():
    numChance = input('Enter the number of guesses you would like to take: ')
    try:
        numChance = int(numChance)
    except:
        print ('Enter a valid number')
    return numChance

def main():
    total_guesses = numGuesses()
    while count < total_guesses:
        get_input()
        print ("\nNumber of guesses left: ", total_guesses - count)
    print ("Oops!!! You have exhausted the number of guesses!!")

if __name__ == "__main__":
    main()
