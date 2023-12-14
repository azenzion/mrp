# Import the draft data
import itertools
import os
import csv
import time
import pickle
from statistics import mean
import numpy as np
import statsmodels.api as sm
import math
import pandas as pd
import matplotlib.pyplot as plt


csv_file_path = os.path.join(os.path.dirname(__file__),
                             "..",
                             "data",
                             "draft_data_public.LTR.PremierDraft.csv")
cardlist_file_path = os.path.join(os.path.dirname(__file__),
                                  "..",
                                  "data",
                                  "ltr_cards.txt")

debug = False

num_drafts = 10000

OPTIMIZE_STORAGE = True

cardNamesHash = {}
cardNumsHash = {}

# Harcoded list of red removal spells
redRemoval = ["Fear, Fire, Foes!",
              "Smite the Deathless",
              "Improvised Club",
              "Foray of Orcs",
              "Cast into the Fire",
              "Breaking of the Fellowship",
              "Spiteful Banditry"]


def get_colours():
    COLOURS = ['W', 'U', 'B', 'R', 'G', '']

    # Generate a list of all possible colour combinations
    temp_colours = []
    for i in range(1, len(COLOURS) + 1):
        temp_colours.extend(list(itertools.permutations(COLOURS, i)))

    return COLOURS


COLOURS = get_colours()

emptyColoursDict = {}
for colour in COLOURS:
    emptyColoursDict[colour] = 0

# Magic numbers
NUM_HEADERS = 12
END_OF_PACK = 277
END_OF_POOL = 2 * END_OF_PACK - NUM_HEADERS - 1

# Dawn of a new age is at the end for some reason
DAWN_OF_A_NEW_HOPE_PACK = END_OF_POOL + 1
DAWN_OF_A_NEW_HOPE_POOL = END_OF_POOL + 2


# Wrapper for picks
class Draft:

    def __init__(self):
        self.draft_id = ""
        self.picks = []


# A draft pick
class Pick:

    def __init__(self):

        self.draft_id = 0
        self.pack_number = 0
        self.pick_number = 0
        self.pick = 0
        self.pack_cards = []

        if not OPTIMIZE_STORAGE:
            self.expansion = ""
            self.event_type = ""
            self.draft_time = 0
            self.rank = 0
            self.event_match_wins = 0
            self.event_match_losses = 0
            self.pick_maindeck_rate = 0.0
            self.pick_sideboard_in_rate = 0.0
            self.pool_cards = []

        # We set these later
        self.numCardInPool = {}
        self.colourInPool = {}


# card
class Card:

    def __init__(self):
        self.name = ""
        self.colour = ""
        self.rarity = ""
        self.numberSeen = 0
        self.alsa = 0.0
        self.numberPicked = 0
        self.avgTakenAt = 0.0
        self.gamesPlayed = 0
        self.gamesPlayedWinrate = 0.0
        self.openingHand = 0.0
        self.openingHandWinrate = 0.0
        self.gamesDrawn = 0
        self.gamesDrawnWinrate = 0.0
        self.gameInHand = 0
        self.gameInHandWinrate = 0.0
        self.gamesNotSeen = 0
        self.gamesNotSeenWinrate = 0.0
        self.improvementWhenDrawn = 0.0

        self.substitutes = []
        self.timesSeen = 0
        self.timesPicked = 0
        self.picks = []
        self.pairwisePickRate = {}

    # Do sorts by name
    def __lt__(self, other):
        return self.name < other.name


def get_drafts_from_cache():
    # Read in the draft data from the cache
    with open(os.path.join(os.path.dirname(__file__),
                           "..",
                           "data",
                           "drafts.pickle"),
              "rb") as f:
        return pickle.load(f)


def name_to_card(card_name, card_data):
    if card_name in card_data:
        return card_data[card_name]

    print("ERROR: Card not found in card_data")
    print(f"Card: {card_name}")
    exit(1)


# Return card data for two cards
# If the input is a tuple, it is assumed to be a pair
# If the input is a string, it is assumed to be a pair of the form "card1 & card2"
def get_cards_from_pair(pair, card_data) -> tuple:
    if type(pair) is tuple:
        card1_name = pair[0]
        card2_name = pair[1]
    else:
        card1_name = pair.split(" & ")[0]
        card2_name = pair.split(" & ")[1]

    card1 = name_to_card(card1_name, card_data)
    card2 = name_to_card(card2_name, card_data)

    if not card1 or not card2:
        print("ERROR: Card not found in card_data")
        print(f"Card 1: {card1_name}")
        print(f"Card 2: {card2_name}")
        exit(1)

    return card1, card2


# In the 17lands data, percentages are stored as strings
# This function parses them into floats
def parse_percentage(percentage):
    if percentage == "":
        return 0.0
    else:
        if 'pp' in percentage:
            return float(percentage[:-2])
        else:
            return float(percentage[:-1])


def compute_pairwise_pickrate(drafts, card1, card2):

    # Look for drafts where a player chose between card1 and card2
    choice_count = 0
    card1_count = 0
    card2_count = 0

    # Get the set number of each card
    # This is how the cards are stored as a memory optimization
    card1_num = cardNumsHash[card1]
    card2_num = cardNumsHash[card2]

    # Create the list of all picks
    all_picks = []
    for draft in drafts:
        all_picks.extend(draft.picks)

    # picks
    for pick in all_picks:
        if card1_num in pick.pack_cards and card2_num in pick.pack_cards:

            # Check if the player picked one of the two cards
            if pick.pick == card1_num or pick.pick == card2_num:
                choice_count += 1

            # Check whether they picked card1 or card2
            if pick.pick == card1_num:
                card1_count += 1
            elif pick.pick == card2_num:
                card2_count += 1

    # If there are no picks with the choice, return 0
    if choice_count == 0:
        return 0, 0

    if debug:
        # Print the number of drafts with the choice
        print(f"Number of picks with the choice: {choice_count}")

        print(f"Number of times they picked {card1}: {card1_count}")
        print(f"Number of times they picked {card2}: {card2_count}")

        # Print percentage players chose each card
        print(f"Percentage of times they picked {card1}:'")
        print(f"{card1_count / choice_count}")
        print(f"Percentage of times they picked {card2}:'")
        print(f"{card2_count / choice_count}")

    return card1_count / choice_count, card2_count / choice_count


def parse_drafts(csv_file_path, ltr_cards, numDrafts=1000):
    csv_reader = None
    print('begin parsing draft data')
    with open(csv_file_path, "r") as f:
        # Create a CSV reader
        csv_reader = csv.reader(f, delimiter=",")

        # Read the header row
        header_row = next(csv_reader)
        if debug:
            print(header_row)

        # Parse into draft objects
        drafts = []

        # Create fist empty draft
        draft = Draft()
        drafts.append(draft)

        current_draft = draft

        # Set the first draft id
        first_data = next(csv_reader)

        current_draft.draft_id = first_data[2]

        # walk back the reader so the loop starts at the first data row
        # go to start of file
        f.seek(0)
        # skip header row
        next(csv_reader)

        if debug:
            print("1st Draft ID: " + current_draft.draft_id)

        for row in csv_reader:

            # Make into a pick object
            pick = Pick()
            pick.draft_id = row[2]
            pick.pack_number = row[7]
            pick.pick_number = row[8]
            pick.pick = cardNumsHash[row[9]]

            if not OPTIMIZE_STORAGE:
                pick.expansion = row[0]
                pick.event_type = row[1]
                pick.draft_time = row[3]
                pick.rank = row[4]
                pick.event_match_wins = row[5]
                pick.event_match_losses = row[6]
                pick.pick_maindeck_rate = row[10]
                pick.pick_sideboard_in_rate = row[11]

            # Parse the pack
            for i in range(NUM_HEADERS, END_OF_PACK):
                if row[i] == "1":
                    pick.pack_cards.append(i - NUM_HEADERS)
            # Dawn of a new hope
            # Idiosyncratic, but it's at the end of the pack
            if row[DAWN_OF_A_NEW_HOPE_PACK] == "1":
                pick.pack_cards.append(265)

            # Parse the pool
            if not OPTIMIZE_STORAGE:
                for i in range(END_OF_PACK, END_OF_POOL):
                    num_card_in_pool = row[i]
                    for k in range(int(num_card_in_pool)):
                        pick.pool_cards.append(i - END_OF_PACK)
                # Dawn of a new hope
                if row[DAWN_OF_A_NEW_HOPE_POOL] == "1":
                    pick.pool_cards.append(265)

            if pick.draft_id == current_draft.draft_id:
                current_draft.picks.append(pick)

            else:

                # Sort on pick number before moving on
                current_draft.picks.sort(key=lambda x: x.pick_number)

                draft = Draft()
                draft.draft_id = pick.draft_id
                drafts.append(draft)
                current_draft = draft
                current_draft.picks.append(pick)

            # Stop after num_drafts drafts
            if len(drafts) > numDrafts:
                break

    return drafts


def find_inversion_pairs(pairs, card_data, only_return_inverted=True) -> dict:
    inversion_pairs = {}

    card1_pickrate = 0
    card2_pickrate = 0

    # Get all pairs and their pick rates
    for pair in pairs.keys():
        card1, card2 = get_cards_from_pair(pair, card_data)
        card1_name = card1.name
        card2_name = card2.name

        print(f"Computing inversion for {card1_name} and {card2_name}")

        card1_pickrate = pairs[pair][0]
        card2_pickrate = pairs[pair][1]

        print(f"{card1_name} pickrate: {card1_pickrate}")
        print(f"{card2_name} pickrate: {card2_pickrate}")

        # Also get the GIH winrate for each card
        card1_gih_winrate = card1.gameInHandWinrate
        card2_gih_winrate = card2.gameInHandWinrate

        # Find the higher and lower winrate cards
        higher_winrate_card = None
        if card1_gih_winrate > card2_gih_winrate:
            higher_winrate_card = card1
            higher_winrate = card1_gih_winrate
            lower_winrate = card2_gih_winrate
        else:
            higher_winrate_card = card2
            higher_winrate = card2_gih_winrate
            lower_winrate = card1_gih_winrate

        # Determine the more picked card
        more_picked_card = None
        if card1_pickrate > card2_pickrate:
            more_picked_card = card1
            higher_pickrate = card1_pickrate
            lower_pickrate = card2_pickrate
        else:
            more_picked_card = card2
            higher_pickrate = card2_pickrate
            lower_pickrate = card1_pickrate

        # Continue if the more picked card has a higher GIH winrate
        if only_return_inverted and more_picked_card == higher_winrate_card:
            continue

        winrate_difference = higher_winrate - lower_winrate
        pickrate_difference = higher_pickrate - lower_pickrate

        inversion = winrate_difference * pickrate_difference

        more_picked_card_name = more_picked_card.name
        less_picked_card_name = higher_winrate_card.name

        # Store the pair and the pick rates
        inversion_pairs[(more_picked_card_name, less_picked_card_name)] = {"card1_pickrate": card1_pickrate,
                                           "card2_pickrate": card2_pickrate,
                                           "card1_gih_winrate": card1_gih_winrate,
                                           "card2_gih_winrate": card2_gih_winrate,
                                           "inversion": inversion}

    # Print the number of inversion pairs
    print(f"Number of inversion pairs: {len(inversion_pairs.keys())}")

    return inversion_pairs


def getCardData():
    card_data = {}

    with open(os.path.join(os.path.dirname(__file__), "..", "data", "ltr-card-ratings-2023-09-17.csv"), "r") as f:
        csv_reader = csv.reader(f, delimiter=",")

        header_row = next(csv_reader)
        if debug:
            print(header_row)

        # Winrates are in the form "50.5%"
        # Remove the % and convert to float
        for row in csv_reader:
            nextCard = Card()
            if debug:
                print(row)        
            nextCard.name = row[0]
            nextCard.colour = row[1]
            nextCard.rarity = row[2]
            nextCard.numberSeen = int(row[3])
            nextCard.alsa = float(row[4])
            nextCard.numberPicked = int(row[5])
            nextCard.avgTakenAt = float(row[6])

            nextCard.gamesPlayed = int(row[7])
            nextCard.gamesPlayedWinrate = parse_percentage(row[8])

            nextCard.openingHand = float(row[9])
            nextCard.openingHandWinrate = parse_percentage(row[10])

            nextCard.gamesDrawn = int(row[11])
            nextCard.gamesDrawnWinrate = parse_percentage(row[12])

            nextCard.gameInHand = int(row[13])
            nextCard.gameInHandWinrate = parse_percentage(row[14])

            nextCard.gamesNotSeen = int(row[15])
            nextCard.gamesNotSeenWinrate = parse_percentage(row[16])

            # These are like "5.5pp"
            nextCard.improvementWhenDrawn = parse_percentage(row[17])

            card_data[nextCard.name] = nextCard

    return card_data

# Returns a list of tuples
def getPairs(card_data):
    pairs = []
    for colour in COLOURS:
        for card in card_data.values():
            if card.colour == colour:
                for otherCard in card_data.values():
                    if otherCard.colour == colour:
                        if card != otherCard:
                            pair = (card, otherCard)
                            pairs.append(pair)

    # Print the number of pairs
    print(f"Number of pairs: {len(pairs)}")  
    return pairs

# Compute the pick rate for each card
def computePickRates(drafts, card_data):
    for draft in drafts:
        for pick in draft.picks:
            for card in pick.pack_cards:
                cardName = card
                card_data[cardName].timesSeen += 1

                # If the card was picked, increment the number of times it was picked
                if card == pick.pick:
                    card_data[cardName].timesPicked += 1
    return card_data

# Compute the pick rate of a card conditioned on having one other card of the same colour in the pool
def pickRateColour(packCardName, drafts, num_cards=1):
    timesSeenWithColour = 0
    timesPickedWithColour = 0

    packCard = name_to_card(packCardName, card_data)
    cardColour = packCard.colour

    for draft in drafts:
        for pick in draft.picks:
            if packCardName not in pick.pack_cards:
                continue

            if packCardName in pick.pack_cards:
                num_colour_cards = 0
                for poolCardName in pick.pool_cards:
                    poolCardData = name_to_card(poolCardName, card_data)
                    poolCardColour = poolCardData.colour
                    if poolCardColour == cardColour:
                        num_colour_cards += 1
                        
                #print(f"Found {num_colour_cards} {cardColour} cards in the pool")
                if num_colour_cards >= num_cards:
                    #print(f'Counting as seen')
                    timesSeenWithColour += 1
                    #print(f"Card picked was {pick.pick}")
                    if pick.pick == packCardName:
                            #print(f'Counting as picked')
                            timesPickedWithColour += 1

    if timesSeenWithColour == 0:
        print(f"Could not find any drafts with {packCard} in the pack and {num_cards} other {cardColour} cards in the pool")
        return 0.0
    
    if debug:
        print(f"Number of times {card1} while at least {num_cards} {cardColour} cards in pool: " + str(timesSeenWithColour))
        print(f"Number of times {card1} picked while at least {num_cards} {cardColour} cards in pool: " + str(timesPickedWithColour))
    
        print (f"Pick rate of {packCardName} with {num_cards} other {cardColour} cards in the pool: {timesPickedWithColour / timesSeenWithColour}")

    return timesPickedWithColour / timesSeenWithColour

# Conditional pick rate, how many times was card2 picked when card1 was in the pool
# num_pool_card is the number of card1 that need to be in the pool
def computeConditionalPickRate(poolCard, packCard, drafts, num_pool_card=1):
    timesSeenWhileInPool = 0
    timesPickedWhileInPool = 0

    picksChoosingPackCard = []
    picksNotChoosingPackCard = []

    # Look for drafts where a player saw card2 while card1 was in their pool
    for draft in drafts:
        for pick in draft.picks:
            if packCard not in pick.pack_cards:
                continue

            # Get the number of poolCard in the pool
            numPoolCards = 0
            for poolCardName in pick.pool_cards:
                if poolCardName == poolCard:
                    numPoolCards += 1
                
                # Also count packCard as a substitute for itself
                #if poolCardName == packCard:
                #    numPoolCards += 1

                
            if numPoolCards >= num_pool_card:
                #print(f"Number of {card1} seen in the pool: {num_pool_card_seen}")
                #print(f"Counting as saw {card2} while {card1} was in the pool")

                timesSeenWhileInPool += 1

                #print(f"Card picked was {pick.pick}")
                
                if pick.pick == packCard:
                    #print(f"Counting as picked {card2} while {card1} was in the pool")
                    timesPickedWhileInPool += 1
                else:
                    #print(f"Counting as did not pick {card2} while {card1} was in the pool")
                    picksNotChoosingPackCard.append(pick)

    if timesSeenWhileInPool == 0:
        if debug:
            print("Could not find any picks with " + packCard + " in the pack and " + str(num_pool_card) + " " + poolCard + " in the pool")
        return 0.0
    
    if debug:
        print(f"Number of times {packCard} seen while {num_pool_card} {poolCard} in pool: " + str(timesSeenWhileInPool))
        print(f"Number of times {packCard} picked while {num_pool_card} {poolCard} in pool: " + str(timesPickedWhileInPool))

        # Print the picks where the player didn't pick card2
        """
        print(f"Picks where {card2} was not picked while {num_pool_card} {card1} were in the pool")
        for pick in picksNotChoosingPackCard:
            print(f"Pick: {pick.pick}")
            print(f"Pack: {pick.pack_cards}")
            print(f"Pool: {pick.pool_cards}")
        """
            


        print (f"Pick rate of {packCard} while at least {num_pool_card} {poolCard} was in the pool: {timesPickedWhileInPool / timesSeenWhileInPool}")
    return timesPickedWhileInPool / timesSeenWhileInPool

def elasticitySubstitution(card1, card2, drafts):

    # Compute the colour conditional pick rate for the cards
    colourConditionalPickRates = []
    for i in range(1, 20):
        card1ColourPickRate = pickRateColour(card1, drafts, i)
        colourConditionalPickRates.append(card1ColourPickRate)

    # Remove 0s and 1s from the list
    colourConditionalPickRates = [x for x in colourConditionalPickRates if x != 0.0 and x != 1.0]
    
    card1ColourPickRate = mean(colourConditionalPickRates)
    #card2ColourPickRate = pickRateColour(card2, drafts)

    # Compute the card conditional pick rate
    cardConditionalPickrates = []
    for i in range(1, 20):
        card1ConditionalPickRate = computeConditionalPickRate(card2, card1, drafts, i)
        cardConditionalPickrates.append(card1ConditionalPickRate)

    # Remove 0s and 1s from the list
    cardConditionalPickrates = [x for x in cardConditionalPickrates if x != 0.0 and x != 1.0]
    if len(cardConditionalPickrates) == 0:
        print(f"Could not compute conditional pick rate for {card1} and {card2}")
        return None
    card1ConditionalPickRate = mean(cardConditionalPickrates)
    #card2ConditionalPickRate = computeConditionalPickRate(card1, card2, drafts)

    # Compute the elasticity of substitution
    #elasticity = (card1ConditionalPickRate - card1ColourPickRate) / (card2ConditionalPickRate - card2ColourPickRate)
    elasticity = (card1ConditionalPickRate - card1ColourPickRate)

    # Compute the elasticity for the other two
    #elasticity2 = card2ConditionalPickRate - card2ColourPickRate    

    if debug:
        print(f"Elasticity of substitution between {card1} and {card2}: {elasticity}")

    #print(f"Elasticity of substitution between {card2} and {card1}: {elasticity2}")
    return elasticity


def parsePoolInfo(cardNames, drafts):

    print(f"Computing draft stats for {cardNames}")

    for draft in drafts:

        # Track the pool as we go
        draftPool = []

        for pick in draft.picks:
            pickName = cardNamesHash[pick.pick]

            # initialize values
            for cardName in cardNames:
                # Find the key which has value cardName
                # in cardNamesHash
                cardNum = cardNumsHash[cardName]
                if cardNum not in pick.numCardInPool:
                        pick.numCardInPool[cardNum] = 0

            # Pool analysis
            # Count number of coloured cards in the pool
            for poolCardName in draftPool:
                for cardName in cardNames:
                    if cardName == poolCardName:
                        cardNum = cardNumsHash[cardName]
                        pick.numCardInPool[cardNum] += 1

                poolCard = name_to_card(poolCardName, card_data)
                poolCardColour = poolCard.colour

                if poolCardColour not in pick.colourInPool:
                    pick.colourInPool[poolCardColour] = 0
                pick.colourInPool[poolCardColour] += 1
            
            for cardName in cardNames:
                card = name_to_card(cardName, card_data)
                cardNum = cardNumsHash[cardName]
                
                if cardNum in pick.pack_cards:
                    
                    # Store with each card a list of picks where the card was seen
                    card.picks.append(pick)

                    card.timesSeen += 1
                    if cardName == pick.pick:
                        card.timesPicked += 1

            # Add the pick to the pool
            draftPool.append(pickName)

    return card_data


def compareSubstitutes(card1, card2, cardList, card_data):

    # Compute the substitutes for a card within a list of cards
    # hang on to the complements even though we don't use them at the moment
    card1Subs, card1Comps = findTopSubstitutes(card1, redCards)
    card2Subs, card2Comps = findTopSubstitutes(card2, redCards)

    # print the total number of substitutes for each
    print(f"Number of substitutes for card1: {len(card1Subs)}")
    print(f"Number of substitutes for card2: {len(card2Subs)}")

    # Eliminate the substitutes with substitution elasticity > -0.005
    # This is to eliminate cards that are not substitutes
    substitutionThreshold = -0.005

    # Eliminate the substitutes with elasticity > -0.005

    #card1Subs = [x for x in rallySubs if x[1] < -0.005]
    #card2Subs = [x for x in smiteSubs if x[1] < -0.005]

    eliminateRares = False

    if eliminateRares:
        # Remove rares and mythic rares
        card1Subs = [x for x in card1Subs if card_data[x[0]].rarity != "R" and card_data[x[0]].rarity != "M"]
        card2Subs = [x for x in card2Subs if card_data[x[0]].rarity != "R" and card_data[x[0]].rarity != "M"]

    # Print the number of substitutes after eliminating
    print(f"Number of substitutes for {card1} after eliminating: {len(card1Subs)}")
    print(f"Number of substitutes for {card2} after eliminating: {len(card2Subs)}")

    print("=====================================")

    # Print each substitute and its elasticity
    print("{card1} substitutes")
    for substitute in card1Subs:
        print(f"{substitute[0]}: {substitute[1]}")

    print("=====================================")
    print("{card2} substitutes")
    for substitute in card2Subs:
        print(f"{substitute[0]}: {substitute[1]}")

    print("=====================================")
    print("Total observations of substitutes")

    card1SubsCount = 0 
    card1SubsWeighted = 0
    for card1Sub in card2Subs:
        card1SubName = card1Sub[0]
        card1SubElas = card1Sub[1]

        card1SubsCount += card_data[card1SubName].numberSeen
        card1SubsWeighted += card_data[card1SubName].numberSeen * card1SubElas

    card2SubsCount = 0
    card2SubsWeighted = 0
    for card2Sub in card2Subs:
        card2SubName = card2Sub[0]
        card2SubElas = card2Sub[1]

        card2SubsCount += card_data[card2SubName].numberSeen
        card2SubsWeighted += card_data[card2SubName].numberSeen * card2SubElas

    print(f"Number of times you see {card1} substitutes: {card1SubsCount}")
    print(f"Number of times you see {card2} substitutes: {card2SubsCount}")

    print(f"Weighted number of times you see {card1} substitutes: {card1SubsWeighted}")
    print(f"Weighted number of times you see {card2} substitutes: {card2SubsWeighted}")


def checkIfSubstitutes(results, numControls, labels):

    indexOfOneSubstitute = numControls
    indexOfTwoSubstitutes = numControls + 1

    elasticity = 0.0

    # Total number of observations
    totalNobs = 0


    # Number of observations for each number of substitutes
    # This is the number that we want out of nObs
    j = 0
    for i in range(indexOfOneSubstitute, len(results.params) - 1):
        j += 1
        if debug:
            print(f"Calculating {labels[indexOfTwoSubstitutes]} - {labels[indexOfTwoSubstitutes]}")

        nextIdx = i + 1

        elasticity = results.params[nextIdx] - results.params[indexOfOneSubstitute]

        #multiply by the number of observations at that number

        elasticity *= results.nobs[j]

        totalNobs += results.nobs[j]

        
    if totalNobs == 0:
        if debug:
            print('Not enough observations to calculate elasticity of substitution')
        return False, 999
    
    # divide by totalNobs
    elasticity /= totalNobs
        
    if debug:
        print(f"Elasticity of substitution: {elasticity}")

    if elasticity < 0:
        return True, elasticity
    else:
        return False, elasticity


def findTopSubstitutes(cardName, cards):
    # Count the number of substitutes for Rally at the Hornburg and Smite the Deathless
    substitutes = []
    complements = []

    for card in cards:
        elasticity = checkSubstitution(card, cardName, num_multiples=10, checkCard1Colour=True, checkCard2Colour=False)

        if elasticity < 0:
            substitutes.append((card, elasticity))
        else:
            complements.append((card, elasticity))

    print(f"Number of substitutes for {cardName}: {len(substitutes)}")

    # Print what the substitutes are and their elasticity
    # Sorted by their elasticity
    substitutes = sorted(substitutes, key=lambda x: x[1], reverse=True)

    for substitute in substitutes:
        print(f"{substitute[0]}: {substitute[1]}")

    return substitutes, complements


# for picks where poth cards were present
# compute the rate at which card 1 was picked
# and the rate at which card 2 was picked
# should sum to 1
def compute_pairwise_pickrates(pairs, drafts):
    pairs_with_pickrates = {}
    # Compute the pairwise pick rate for each pair
    for card1, card2 in pairs:
        pickrate = compute_pairwise_pickrate(drafts, card1, card2)
        pairs_with_pickrates[card1, card2] = pickrate

    return pairs_with_pickrates


def checkSubstitution(card1, substitutesList, num_multiples=10, checkCard1Colour=True, checkCard2Colour=False):
    if debug:
        print(f"Checking substitution for {card1}, among substitutes {substitutesList}")

    # Allow function to take a single substitute
    if type(substitutesList) == str:
        substitutesList = [substitutesList]

    # Keep a running total for indexes
    numControls = 0
    # Constant term
    numControls += 1
    # number of card1 in pool
    numControls += 1

    card1Obj = name_to_card(card1, card_data)
    card1Colour = card1Obj.colour

    card2Obj = None
    card2Colour = None

    if checkCard1Colour:
        numControls += 1

    if checkCard2Colour:
        card2Obj = name_to_card(substitutesList[0], card_data)
        card2Colour = card2Obj.colour

        if card1Colour == card2Colour:
            checkCard2Colour = False

            if debug:
                print(f"Card 1 and card 2 are the same colour: {card1Colour}")
                print("Will not check for card 2 colour")
        else:
            numControls += 1

    # Remove card1 from the list of substitutes
    # This is so we don't accidentally double count
    if card1 in substitutesList:
        substitutesList.remove(card1)

    y = []
    card1ColourCount = []
    card1InPool = []

    card2ColourCount = []
    numSubstitutes = {}

    for i in range(num_multiples + 1):
        numSubstitutes[i] = []

    for pick in card1Obj.picks:
        pickName = cardNamesHash[pick.pick]

        if pickName == card1:
            y.append(1)
        else:
            y.append(0)

        # Append number of cards that are the same colour as card1
        if checkCard1Colour:
            card1ColourTotal = 0
            for colourOfCard in pick.colourInPool.keys():
                if card1Colour in colourOfCard:
                    card1ColourTotal += pick.colourInPool[colourOfCard]

            # Take the log of the number of cards
            # This is to account for diminishing returns
            card1ColourTotal = math.log(card1ColourTotal + 1)
            card1ColourCount.append(card1ColourTotal)

        if checkCard2Colour:
            card2ColourTotal = 0
            for colourOfCard in pick.colourInPool.keys():
                if card2Colour in colourOfCard:
                    card2ColourTotal += pick.colourInPool[colourOfCard]
            # Take the log of the number of cards
            # This is to account for diminishing returns
            card2ColourTotal = math.log(card2ColourTotal + 1)
            card2ColourCount.append(card2ColourTotal)

        substitutesCount = 0
        # Count total number of substitutes in the pool
        for cardName in substitutesList:
            cardNum = cardNumsHash[cardName]
            if cardNum in pick.numCardInPool:
                substitutesCount += pick.numCardInPool[cardNum]
                break

        for i in numSubstitutes.keys():
            if substitutesCount == i:
                numSubstitutes[i].append(1)
            else:
                numSubstitutes[i].append(0)

        # Control for the number of card1 in the pool
        card1Num = cardNumsHash[card1]
        if card1Num in pick.numCardInPool:
            card1InPool.append(pick.numCardInPool[card1Num])
        else:
            card1InPool.append(0)
        

    # Eliminate values that didn't have enough observations
    threshold = 40
    temp = {}
    numObservations = []
    for i in range(1, num_multiples + 1):
        totalObservations = sum(numSubstitutes[i])
        numObservations.append(totalObservations)

        if totalObservations < threshold:
            if debug:
                print(f"Not enough observations with {i} cards in the pool.")
                print("Will eliminate all higher values")
            break
        else:
            temp[i] = numSubstitutes[i]

    numSubstitutes = temp

    endog = np.array(y)
    
    # Assemble exog matrix
    exog = None

    if checkCard1Colour:
        exog = np.array(card1ColourCount)

        if checkCard2Colour:
            exog = np.column_stack((exog, card2ColourCount))

        exog = np.column_stack((exog, card1InPool))
    else:
        exog = np.array(card1InPool)

    # Add the number of substitutes in the pool
    for i in numSubstitutes.keys():
        exog = np.column_stack((exog, numSubstitutes[i]))

    # Add a constant term
    exog = sm.add_constant(exog)

    model = sm.OLS(endog, exog)

    # Label the variables in the model
    labels = ["Constant"]
    if checkCard1Colour:
        labels.append(f"{card1Colour} cards in pool")
    if checkCard2Colour:
        labels.append(f"{card2Colour} cards in pool")

    labels.append(f"{card1} in pool")

    if len(substitutesList) == 1:
        for i in numSubstitutes.keys():
            labels.append(f"{i} {substitutesList[0]} in pool")
    else: 
        for i in numSubstitutes.keys():
            labels.append(f"{i} substitutes in pool")

    model.exog_names[:] = labels

    results = model.fit()

    # append total observations at each number to the results
    # We use this to calculate our elasticity of substitution
    results.nobs = numObservations

    substitutes, elasticity = checkIfSubstitutes(results, numControls, labels)

    if debug:
        if substitutes:
            print(f"{card1} and {substitutesList} are substitutes")
        else:
            print(f"{card1} and {substitutesList} are not substitutes")

    return elasticity


def regressOnNumCard2(card1, card2, drafts, card_data):

    #card_data = computeDraftStats(card1, card2, drafts)

    y = []
    colours = []

    card2InPool = {}

    # Max number of card2 multiples to consider
    num_multiples = 5
    for i in range(0, num_multiples + 1):
        card2InPool[i] = []


    for pick in card_data[card1].picks:
        if pick.pick == card1:
            y.append(1)
        else:
            y.append(0)
        
        colours.append(pick.colourInPool)

        numCard2 = pick.numCardInPool[card2]
        
        for i in card2InPool.keys():
            if numCard2 == i:
                card2InPool[i].append(1)
            else:
                card2InPool[i].append(0)

        

    # Eliminate any number of card2 with less than 30 samples
    tempCard2InPool = {}
    for i in range(0, num_multiples + 1):
        if sum(card2InPool[i]) < 100:
            print(f"Eliminating {i} {card2} from the regression due to not enough samples")
            num_multiples -= 1
        else:
            tempCard2InPool[i] = card2InPool[i]
    card2InPool = tempCard2InPool

    # Do a regression of Y on:
    # colours
    # an indicator variable for x = 0 to num_multiples, indicating whethere there are that many card2 in the pack

    endog = np.array(y)

    # Assemble exog matrix

    # maybe control for number of same colour cards in pack but seems not to matter
    # This lets us differentiate between drafts where you have 0 card1 because:
    # - You are not in a deck that can play card1
    # vs
    # - You are in a deck that can play card1 but you didn't see any
    # exog = np.array(colours)

    exog = np.array(card2InPool[0])
    for i in range(1, num_multiples + 1):
        try:
            exog = np.column_stack((exog, card2InPool[i]))
        except KeyError as e:
            print(f"KeyError: {e}")

    # Add a constant term
    exog = sm.add_constant(exog)

    model = sm.OLS(endog, exog)

    results = model.fit()

    weightedAverage = 0.0
    #print(results.summary())
    try:
        # A card is said to be a substitute if the coefficient on number of cards is negative for n>=1
        # and the coefficient on number of cards is positive for n=0
        if debug:
            # Print the coefficients
            print(f"Effects of {card2} in pool on picking {card1}")
            for i in range(0, num_multiples + 1):
                print(f"{i}: {results.params[i + 1]}")

        # Compute the weighted average of the nonzero coefficients
        for i in range(1, num_multiples + 1):
            num_picks = sum(card2InPool[i])
            weightedAverage += results.params[i + 1] * num_picks
    except IndexError as e:
        print(f"IndexError: {e}")
    except KeyError as e:
        print(f"KeyError: {e}")

    # Weighted average of the coefficients for 1 to num_multiples must be negative
    # Then we call the cards substitutes 
       
    substitute = weightedAverage < 0.0

    print(f"Weighted average of coefficients: {weightedAverage}")

    # All coefficients must be negative
    # substitute = all(results.params[1:] < 0.0)
    if substitute:
        print(f"{card2} is a substitute for {card1}")
    else:
        print(f"{card2} is not a substitute for {card1}")

    return substitute




# Create initial timestamp
timestamp = time.time()

ltr_cards = []
with open(cardlist_file_path, "r") as f:
    for line in f:
        ltr_cards.append(line.strip())

NUM_CARDS_IN_SET = 266

# Populate the hash of number to card name
# This is to optimize the size of Pick objects
for i in range(NUM_CARDS_IN_SET):
    cardNamesHash[i] = ltr_cards[i]
    cardNumsHash[ltr_cards[i]] = i

GET_DRAFTS_FROM_CACHE = True

drafts = []

# Take a timestamp
timestamp = time.time()

if GET_DRAFTS_FROM_CACHE:
    print("Reading drafts from cache")
    drafts = get_drafts_from_cache()
else:
    drafts = parse_drafts(csv_file_path, ltr_cards, num_drafts)

    # Use pickle to cache
    with open(os.path.join(os.path.dirname(__file__), "..", "data", "drafts.pickle"), "wb") as f:
        pickle.dump(drafts, f)

# Print time to parse drafts
print(f"Time to parse drafts: {time.time() - timestamp}")

card_data = {}
card_data = getCardData()

# Get red cards
redCards = []
for card in card_data.values():
    if card.colour == "R" and not card.rarity == "M" and not card.rarity == "R":
        redCards.append(card.name)

redPairs = []
for card1 in redCards:
    for card2 in redCards:
        if card1 != card2 and (card2, card1) not in redPairs:
            redPairs.append((card1, card2))

# Compute the number of pairs
numPairs = len(redPairs)
print(f"Number of pairs: {numPairs}")

timestamp = time.time()
parsePoolInfo(redCards, drafts)
print(f"Time to parse pool info: {time.time() - timestamp}")

# For each pair, when each card is on offer, compute the pick rate of each card
pairs = compute_pairwise_pickrates(redPairs, drafts)

# Print all the pairwaise pick rates
for pair in pairs.keys():
    print(f"{pair}: {pairs[pair]}")


# Find pairs where the card with the lower GIH winrate is picked more often
inversionPairs = find_inversion_pairs(pairs,
                                      card_data,
                                      only_return_inverted=False)

# Print the inversion pairs
for pair in inversionPairs.keys():
    print(f"{pair}: {inversionPairs[pair]}")
    value = inversionPairs[pair]
    print(f"Card 1 GIH winrate: {value['card1_gih_winrate']}")
    print(f"Card 2 GIH winrate: {value['card2_gih_winrate']}")
    print(f"Card 1 pick rate: {value['card1_pickrate']}")
    print(f"Card 2 pick rate: {value['card2_pickrate']}")
    print(f"Inversion: {value['inversion']}")
    print("=====================================")


# For each red card, compute its substitutes
# This is all the cards with elasticity of substitution < -0.005
# This is a list of tuples
# The first element is the card name
# The second element is the elasticity of substitution

# If there is a local cache, read from that
# Otherwise, compute the substitutes

if os.path.exists(os.path.join(os.path.dirname(__file__), "..", "data", "substitutes.pickle")):
    with open(os.path.join(os.path.dirname(__file__), "..", "data", "substitutes.pickle"), "rb") as f:
        redCardsWithSubstitutes = pickle.load(f)
else:
    redCardsWithSubstitutes = {}
    for card in redCards:
        print(f"Computing substitutes for {card}")
        substitutes = []
        for card2 in redCards:
            if card2 != card:
                elasticity = checkSubstitution(card, card2, num_multiples=10, checkCard1Colour=True, checkCard2Colour=False)
                if elasticity < -0.005:
                    substitutes.append((card2, elasticity))

        redCardsWithSubstitutes[card] = substitutes

    # Cache the substitutes
    with open(os.path.join(os.path.dirname(__file__), "..", "data", "substitutes.pickle"), "wb") as f:
        pickle.dump(redCardsWithSubstitutes, f)

# Print the red cards with their substitutes
for card in redCardsWithSubstitutes.keys():
    print(f"{card}: {redCardsWithSubstitutes[card]}")

# For each card, calculate the number of times it was seen
# append this info to card_data
for card in redCards:
    cardSubstitutes = redCardsWithSubstitutes[card]
    cardSubstitutesCount = 0
    for card2, elasticity in cardSubstitutes:
        cardSubstitutesCount += card_data[card2].numberSeen

    card_data[card].numberSubstitutes = cardSubstitutesCount


# For each inversion pair, calculate the difference in substitutes seen
# Also calculate the difference in pick rate between the two cards
# Regress the difference in pick rate on the difference in substitutes seen
# Create a list of tuples
# Each tuple is a pair of cards
# The first card is the more picked card
# The second card is the less picked card
# The third element is the difference in pick rate between the two cards
# The fourth element is the difference in substitutes seen between the cards
regressionData = []
for pair in inversionPairs.keys():
    card1, card2 = get_cards_from_pair(pair, card_data)

    if type(card1) is Card:
        card1 = card1.name
    if type(card2) is Card:
        card2 = card2.name

    print(f"Computing regression data for {card1} and {card2}")

    # Get the coefficient of inversion
    inversion = inversionPairs[pair]["inversion"]

    # For each card, get its substitutes as calculated above
    # Compare the total number of observations of the substitutes
    card1_subs = redCardsWithSubstitutes[card1]
    card2_subs = redCardsWithSubstitutes[card2]

    # Count the total number of observations of the substitutes
    card1_subs_count = card_data[card1].numberSubstitutes
    card2_subs_count = card_data[card2].numberSubstitutes

    # At this point, card1 is the more picked card
    # because of how find_inversion_pairs works

    # We want the difference between substitutes seen for the more picked card
    # and the less picked card
    subs_count_diff = card1_subs_count - card2_subs_count

    # Skip elements where the difference in substitutes seen is 0
    if subs_count_diff == 0:
        print(f"Skipping {card1} and {card2} because the difference in substitutes seen is 0")
        continue

    # Append the tuple to the list
    regressionData.append((card1,
                           card2,
                           inversion,
                           subs_count_diff))

dependent_var = "Coefficient of Inversion"
independent_var = "Substitutes Seen Diff"

# Print the regression data
for i in range(0, len(regressionData)):
    print(f"{regressionData[i][0]} & {regressionData[i][1]}")
    print(f"{dependent_var}: {regressionData[i][2]}")
    print(f"{independent_var}: {regressionData[i][3]}")
    print("=====================================")

# Create a dataframe from the regression data
df = pd.DataFrame(regressionData, columns=["Card 1",
                                           "Card 2",
                                           dependent_var,
                                           independent_var])

# Print the dataframe
print(df)

# Regress the pick rate diff on the subs count diff

# Create the model
model = sm.OLS.from_formula(f"Q('{dependent_var}') ~ Q('{independent_var}')",
                            data=df)

# Fit the model
results = model.fit()

# Print the results
print(results.summary())

# Graph the results
fig = plt.figure(figsize=(12, 8))
fig = sm.graphics.plot_partregress_grid(results, fig=fig)
fig.savefig(os.path.join(os.path.dirname(__file__), "..", "data", "regression.png"))

# Show the plot
plt.show()

# Print the time that script took to run
print("Time to run: " + str(time.time() - timestamp))



exit()




# Compute the number of pairs where card2 is a substitute for card1
numSubstitutes = 0
for i in range(0, len(redCards)):
    for j in range(i + 1, len(redCards)):
        card1 = redCards[i]
        card2 = redCards[j]
        if regressOnNumCard2(card1, card2, drafts, card_data):
            numSubstitutes += 1
            card1 = name_to_card(card1, card_data)
            card1.substitutes.append(card2)

print(f"Number of substitutes: {numSubstitutes}")

# Sort the red cards by number of substitutes
redCardSubstitutes = {}
for card in card_data.values():
    if card.colour == "R":
        redCardSubstitutes[card.name] = len(card.substitutes)

# Sort the cards by number of substitutes
sortedSubstitutes = sorted(redCardSubstitutes.items(), key=lambda x: x[1], reverse=True)

# Print each card, its number of substitutes, and its substitutes
for cardName in sortedSubstitutes:
    card = name_to_card(cardName[0], card_data)
    print(f"{cardName[0]}: {cardName[1]}")
    for substitute in card.substitutes:
        print(f"    {substitute.name}")


# Print the time that script took to run
print("Time to run: " + str(time.time() - timestamp))