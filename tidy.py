#!/usr/bin/env python3
"""
Text Cleaning and Preprocessing Utility

This module provides comprehensive text cleaning and preprocessing functions
for relative clause extraction. It handles various text formatting issues,
punctuation errors, and prepares text for syntactic analysis.

Key Features:
- Punctuation correction and spacing
- Emoji removal and text normalization
- Sentence boundary detection
- Text cleaning for NLP processing

Usage:
    python tidy.py input_file.txt

Dependencies:
    - nltk
    - emoji
    - re (built-in)
    - string (built-in)
"""

import sys
import re
import string
import emoji
# Run "pip install clean-text" on your command line interpreter (e.g. Terminal on Mac)
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize


def main():
    """
    Main function for text cleaning and preprocessing.
    
    Reads input text file, applies cleaning operations, and outputs
    cleaned text suitable for relative clause extraction.
    
    Args:
        sys.argv[1]: Input text file path
    """
    # open .txt file & tokenize it
    filename = str(sys.argv[1])
    file = open(filename, 'rt', encoding="utf-8")
    body = file.read()

    # initialize cleaning up word spacing
    words_init = re.split(' ', body)
    for i, word in enumerate(words_init):
        word_test = list(word)
        if len(word_test) != 0:
            word_test.pop(-1)
        word = word.strip()
        # correct punctuation errors
        if "." in word and (word[-1] != "." or '.' in word_test) and ".)" not in word and ".com" not in word and \
                ".org" not in word and ".net" not in word and ".edu" not in word and "..." not in word and not\
                ("a" or "p" in word.lower() and "m" in word.lower()):
            word = ". ".join(word.split('.'))
            words_init[i] = word
        if "!" in word and word[-1] != "!":
            word = "! ".join(word.split('!'))
            words_init[i] = word
        if "?" in word and word[-1] != "?":
            word = "? ".join(word.split('?'))
            words_init[i] = word
        if "," in word and word[-1] != ",":
            word = ", ".join(word.split(','))
            words_init[i] = word
        if ";" in word and word[-1] != ";":
            word = "; ".join(word.split(';'))
            words_init[i] = word
        has_num = False
        for x in word:
            if x.isnumeric():
                has_num = True
        if ":" in word and word[-1] != ":" and "://" not in word and not has_num:
            word = ": ".join(word.split(':'))
            words_init[i] = word
        if "\'\'" in word:
            word = "\'".join(word.split("\'\'"))
            words_init[i] = word
        if "\"" in word and word[0] != "\"" and word[-1] != "\"":
            word = ".\" ".join(word.split("\""))
            words_init[i] = word

        # Clean up issues with sentences ending without punctuation and merging with other sentences
        # Only works if sentences are still properly capitalized

        # Uncomment the code below if the text file seems to have a lot of sentences running into each other
        # E.g. Hello my name is JackI love playing video gamesIt is fun.
        # if len(word) > 1:
        #     is_cap = 0
        #     alpha = 0
        #     index = 0
        #     more_cap = False
        #     consecutive_cap = False
        #     consecutive_start = 0
        #     consecutive_count = 0
        #     all_cap = False
        #     offenders = []
        #     for x in word:
        #         if (x == "-" and is_cap == 1) or x == " " or x == "\"" or x == ".":
        #             is_cap = 0
        #         if x.isalpha():
        #             alpha += 1
        #         if x.isupper() and x.isalpha():
        #             if not consecutive_cap:
        #                 consecutive_start = index
        #                 consecutive_cap = True
        #                 consecutive_count += 1
        #             elif consecutive_cap:
        #                 consecutive_count += 1
        #             is_cap += 1
        #             if is_cap == 1 and index != 0:
        #                 is_cap += 1
        #             if is_cap >= 2:
        #                 offenders.append(x)
        #                 if is_cap > 2:
        #                     more_cap = True
        #         elif consecutive_cap and index != consecutive_start and consecutive_count == 1:
        #             consecutive_cap = False
        #             consecutive_count = 0
        #         index += 1
        #     if is_cap == alpha or (alpha - is_cap == 1 and "s" in word):
        #         all_cap = True
        #     if is_cap > 1 and not more_cap and not all_cap and offenders[0] != '' and not consecutive_cap:
        #         split_words = word.split(offenders[0])
        #         updated_sentence = split_words[0] + ". " + str(offenders[0]) + split_words[1]
        #         words_init[i] = updated_sentence
        #     elif consecutive_start == 0 and is_cap > 1 and not all_cap and offenders[0] != '':
        #         if consecutive_count != 2:
        #             updated_word = word[0:(consecutive_count - 1)] + ". " + word[(consecutive_count - 1):]
        #         elif len(word) > 2:
        #             updated_word = word[0:2] + ". " + str(word[2]).upper() + word[3:]
        #         else:
        #             updated_word = word[0:2] + "."
        #         words_init[i] = updated_word
        #     elif 1 < is_cap <= 3 and not all_cap and offenders[0] != '':
        #         if consecutive_count != 2:
        #             updated_word = word[0:(consecutive_start + consecutive_cap - 1)] + \
        #                            ". " + word[(consecutive_start + consecutive_count - 1):]
        #         elif (len(word) - consecutive_start) > 2:
        #             updated_word = word[0:(consecutive_start + 2)] + ". " + \
        #                            str(word[(consecutive_start + 2)]).upper() + word[(consecutive_start + 3):]
        #         else:
        #             updated_word = word[0:(consecutive_start + 2)] + "."
        #         words_init[i] = updated_word

        # Fix overcorrection from above
        if "?." in word:
            words_init[i] = word.replace("?.", "?")
        if "!." in word:
            words_init[i] = word.replace("!.", "!")
        if ",." in word:
            words_init[i] = word.replace(",.", ",")

    body = " ".join(words_init)
    file.close()
    body = body.replace(":  .", ": ").replace(": .", ": ").replace("DIT:", "\n.").replace(":. ", ": ")
    lines = sent_tokenize(body)
    # clear out file & prep it for rewriting
    file = open(filename, 'wt', encoding="utf-8")
    file.write("")
    file.close()
    file = open(filename, 'w', encoding="utf-8")

    # initialize variables
    lineset = set()
    print_this = True
    unnecessary = False
    recipe_start = False
    credit = False
    end = False
    title_line = False
    # english = set(string.printable)
    prev_list = False

    # main tidying tasks for each sentence
    for line in lines:
        # remove emojis & excess whitespace from ends
        # line = remove_emoji(line)
        line = line.strip()
        line = line.replace("\n.", ".")
        line = line.replace(":.", ".")
        # Uncomment this only if you need to remove a massive amount of non-english characters
        # line = ''.join(filter(lambda y: y in english or y == " " or y == "\'" or y == "\"", line))

        # clear out unnecessary information (usually formatting info in forums)
        if ("{" in line and "}" not in line) or \
                (("recipe" in line or "Recipe" in line or "ingredients" in line or "Ingredients" in line) and
                 ":" in line) or ("[" in line and "]" not in line):
            unnecessary = True
            recipe_start = True

        # remove excess whitespace from within sentences
        line = line.replace("\t", " ")
        line = line.replace("\n\n\n", "\n\n")
        line = line.replace("  ", " ")
        line = line.replace(" ", " ")

        # remove duplicate sentences
        if not unnecessary and line.strip().lower() in lineset and "thisisbeforeatitle!" != line.strip() and \
                "endofbody!" != line.strip() and "thisisatitle!" != line.strip():
            print_this = False

        # split the sentence into its words
        lwords = re.split(' ', line)

        # get rid of credit/bylines
        if (":" in line and "endofbody!" in line) or "©" in line or "Copyright" in line:
            unnecessary = True
            credit = True

        if print_this and not unnecessary:
            # add sentence to duplicate test bank
            lineset.add(line.strip().lower())
            if "endofbody!" in line and "endofbody!" != line.strip():
                lineset.add(line.replace(" endofbody!", ""))

            # remove any emoticons, bylines in forums, and extra newlines in text
            bracket = False
            words_backwards = lwords.copy()
            words_backwards.reverse()
            new_words = []
            dash = False
            prev_check = False
            for word in words_backwards:
                # letter_count = 0
                # total = 0
                # for x in word:
                #     if x.isalpha() or x == " ":
                #         letter_count += 1
                #     total += 1
                # if total > 0 and (letter_count / total) < (1 / 3):
                #     dash = False
                #     lwords.pop(-1)
                if "]" in word or "}" in word:
                    if ":]" not in word and ":}" not in word:
                        bracket = True
                    dash = False
                    lwords.pop(-1)
                elif "[" in word or "{" in word:
                    if ":[" not in word and ":{" not in word and not bracket and len(new_words) != 0:
                        new_words.clear()
                    dash = False
                    bracket = False
                    lwords.pop(-1)
                elif bracket:
                    dash = False
                    lwords.pop(-1)
                elif len(lwords) == 0 and bracket:
                    dash = False
                    bracket = False
                elif ":)" in word or "(:" in word or " ):" in word or ":(" in word or ":D" in word or " D:" in word \
                        or "\\:" in word or ":\\" in word or "/:" in word or ":/" in word or ";)" in word or "(;" \
                        in word or " );" in word or ";(" in word or ";D" in word or "D;" in word or "\\;" in word \
                        or ";\\" in word or "/;" in word or ";/" in word or ":<" in word or ":>" in word or "<:" in \
                        word or ">:" in word or ";<" in word or ";>" in word or "<;" in word or ">;" in word or ":3" \
                        in word or ";3" in word or "http:" in word or "https:" in word or ".com" in word or ".net" in \
                        word or ".org" in word or ".edu" in word or "www." in word or "o:" in word or "O:" in word or \
                        ":o" in word or ":O" in word or ":0" in word or " 0:" in word or ">" in word or "<" in word:
                    dash = False
                    lwords.pop(-1)
                # elif "--" in word or "__" in word:
                #     dash = False
                #     lwords.pop(-1)
                #     new_words.append(" ")
                elif "said:" in word or "posted:" in word or "wrote:" in word:
                    dash = False
                    lwords.pop(-1)
                    new_words.append("said: ENDOFPOSTLINE.")
                # elif "—" in word:
                #     dash = True
                #     if "post" in word or "said" in word or "wrote" in word:
                #         dash = False
                #         lwords.pop(-1)
                #         new_words.append("said— ENDOFPOSTLINE.")
                elif dash and ("post" in word or "said" in word or "wrote" in word):
                    dash = False
                    lwords.pop(-1)
                    new_words.append("said — ENDOFPOSTLINE.")
                else:
                    dash = False
                    new_words.append(word)
                    lwords.pop(-1)
                # if len(new_words) != 0 and "\n" in new_words[-1] and ".\n" not in new_words[-1]:
                #     new_words[-1] = new_words[-1].replace("\n", ".\n")
                if len(new_words) != 0 and "." in new_words[-1] and new_words[-1][-1] != "." and\
                        new_words[-1][-1] != "\"" and new_words[-1][-1] != "\'":
                    new_words[-1] = new_words[-1].replace(".", ". ")
            lwords = new_words.copy()
            new_words.clear()
            lwords.reverse()

            # place a newline after ends of bodies and titles
            if len(lwords) != 0 and ("thisisatitle!" in lwords[-1] or "endofbody!" in lwords[-1]):
                if "thisisatitle!" in lwords[-1]:
                    title_line = False
                lwords.pop()
                if len(lwords) != 0 and "\n" in lwords[-1]:
                    lwords[-1] = lwords[-1].replace("\n", "")
                sentence = ' '.join(lwords)
                lineset.add(sentence)
                if len(lwords) != 0 and len(lwords) > 4:
                    lwords[-1] = lwords[-1] + "."
                    sentence = sentence + "."
                line = sentence + '\n'
                end = True
            # place a newline before a new title and body
            elif len(lwords) != 0 and ("thisisbeforeatitle!" in lwords[0] or "endofbody!" in lwords[0]):
                if "thisisbeforeatitle!" in lwords[0]:
                    title_line = True
                line = '\n'
            # rewrite spaces between each sentence
            else:
                if len(lwords) != 0 and "." not in lwords[-1] and "!" not in lwords[-1] and "?" not in lwords[-1]:
                    line = ' '.join(lwords) + ". "
                else:
                    line = ' '.join(lwords) + " "

            list_check = search_for_list(line, title_line)
            if ".." in line and "..." not in line:
                line = line.replace("..", ".")

            # don't reprint sentences if they are social promotion, website instructions, lists, or credit lines
            if len(lwords) != 0 and not search_for_social(lwords) and not search_for_extra(lwords) \
                    and not search_for_recipe(lwords, end) and "include the headline" not in line and \
                    not list_check and not short(lwords, title_line) and not forum_check(line):
                file.write(line.replace("  ", " "))
            if list_check and not prev_list:
                file.write("\n")
                prev_check = True
                # Add in full sentences from lists
                if line.strip()[-1] == "." or line.strip()[-1] == "?" or line.strip()[-1] == "!":
                    lines = line.split("\n")
                    if len(lines[-1].strip()) >= 3:
                        file.write(lines[-1])
            elif not list_check:
                prev_check = False
            if len(lwords) == 0 and line == "\n":
                file.write(line)
            prev_list = prev_check

        # if formatting in semicolons is done necessitate processing again
        if "}" in line and "{" not in line or recipe_word(lwords) or credit or ("]" in line and "[" not in line):
            unnecessary = False
        if recipe_start:
            unnecessary = False
            recipe_start = False

        end = False
        print_this = True
    file.close()
    file = open(filename, 'rt', encoding="utf-8")
    body = file.read()
    # remove more emoticons
    body = body.replace("\n\n\n", "\n\n").replace("\n\n\n", "\n\n").replace("\n\n\n", "\n\n")\
        .replace(" !", "!").replace(" .", ".").replace(" ?", "?").replace(" ,", ",")\
        .replace(" :", ":").replace(" ;", ";").replace("{", "(").replace("}", ")").replace("[", "(").replace("]", ")")\
        .replace(": )", "").replace("; )", "")\
        .replace(") ;", "").replace("( :", "").replace("( ;", "").replace(": 0", "").replace("0 :", "")\
        .replace("\n).", "")\
        .replace(": 3", "").replace(": <", "").replace(": >", "").replace("< :", "").replace("> :", "")\
        .replace("\n. ", "")

    # only use this line if you used the capitalization separator above
    # body = body.replace(". s", "s. ")

    file = open(filename, 'wt', encoding="utf-8")
    file.write(body.strip())
    file.close()


# look for social media promotion in the sentence
def search_for_social(array):
    social_counter = 0
    follow = 0
    for x in array:
        y = x.lower()
        if y == "instagram" or y == "twitter" or y == "tiktok" or y == "facebook" \
                or "@" in y or "email" in y or "e-mail" in y or "affiliate" in y or "amazon" in y or "subscribe" in y:
            social_counter += 1
        if y == "follow" or "@" in y or "contact" in y or "share" in y or "buy" in y or "check" in y or "now" in y:
            follow += 1
    if social_counter >= 1 and follow >= 1:
        return True

    return False


# look for extra website-sourced instructions in the file
def search_for_extra(array):
    click_counter = 0
    action = 0
    for x in array:
        y = x.lower()
        if "click" in y or "cookie" in y:
            click_counter += 1
        if "expand" in y or "shrink" in y or "see" in y or "sign" in y or "report" in y or "subscribe" in y \
                or "accept" in y or "here" in y or "link" in y:
            action += 1
        if "more:" in y:
            click_counter += 1
            action += 1
    if click_counter >= 1 and action >= 1:
        return True

    sentence = " ".join(array)
    if "privacy policy" in sentence or "access token" in sentence:
        return True
    if "estimate" in sentence and "read" in sentence and ("minute" in sentence or "hour" in sentence):
        return True
    if "powered by" in sentence or "Powered by" in sentence or "published by" in sentence or "Published by" in sentence:
        return True
    if "You have" in sentence and "ed" in sentence and len(array) <= 6:
        return True
    if "lightbox" in sentence.lower() or "toggle" in sentence.lower() and len(sentence) <= 4:
        return True
    if "please try again" in sentence.lower():
        return True
    count = 0
    for x in sentence:
        if x == ":":
            count += 1
    if ":" in sentence and "\"" in sentence and (len(sentence.split(" ")) <= 5 or count >= 2):
        return True

    return False


# look for recipe ingredient lists
def search_for_recipe(array, end):
    x = array[-1].lower()
    if x == "g." or x == "lb." or x == "oz." or x == "lbs." or x == "tbsp." or x == "tsp.":
        return True
    line = " ".join(array)
    if end and (")" in x or "gram" in line or "cup" in line or "large eggs" in line or "grated" in line
                or "salt" in line):
        return True

    return False


# remove emoji characters
def remove_emoji(sentence):
    return emoji.replace_emoji(sentence, replace='')


# look for recipe-based instruction words
def recipe_word(array):
    for x in array:
        y = x.lower()
        if (y == "desalt" or y == "combine" or y == "clean" or y == "pour" or y == "mix" or y == "boil"
                or y == "peel" or y == "wash" or y == "cut" or y == "slice" or y == "mince" or y == "season" or
                y == "beat" or y == "preheat" or y == "drizzle" or y == "add" or y == "prepare" or y == "remove"
                or y == "freeze" or y == "whisk" or y == "cover" or y == "heat" or y == "chop" or y == "fill"):
            return True

    return False


# look for a list format
def search_for_list(sentence, title):
    count = 0
    prev = ''
    for y in sentence:
        for x in y:
            if x == '\n':
                count += 1
            if prev.isnumeric() and x == ".":
                return True
            prev = x

    if count == 1 and sentence[-1] == "\n" or sentence[0] == "\n":
        return False

    if count >= 1 and not title:
        return True

    # Check for purely numeric sentences
    # every_num = True
    # for x in sentence:
    #     if not x.isnumeric() and x != ".":
    #         every_num = False
    # if every_num:
    #     return True

    return False


# look for short, unnecessary "sentences"
def short(array, title):
    if len(array) <= 4 and not title and "!" not in array[-1] and "." not in array[-1] and "?" not in array[-1]:
        return True
    sentence = ' '.join(array)
    if "details" in array[-1] or "Details" in array[-1] or "related:" in sentence.lower():
        return True
    return False


# look for bylines in a forum-sourced text file
def forum_check(line):
    sentence = line.lower()
    if "said:" in sentence or "posted:" in sentence or "wrote:" in sentence or "said —" in sentence or \
       "posted —" in sentence or "wrote —" in sentence or "from:" in sentence or "edit:" in sentence:
        return True
    return False


if __name__ == "__main__":
    main()
