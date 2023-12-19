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

num_drafts = 100000

# The number of multiples in the pool to consider
# This just needs to be larger than we're likely to see
# in the datav
NUM_MULTIPLES = 10

# For a number x
# If there are less drafts than this with x of card2 in pool
# we will not consider it
OBSERVATIONS_THRESHOLD = 40

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
        card1_name = str(pair[0])
        card2_name = str(pair[1])
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
        csv_reader = csv.reader(f,
                                 delimiter=",")


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


# Takes a list of pairs with their pairwise pick rates
# Returns a dictionary
# Keys are pairs
# More Picked Card, Less Picked Card
# Values are higher pick rate, lower pick rate, higher winrate, lower winrate, inversion
def find_inversion_pairs(pairs, card_data, only_return_inverted=True) -> dict:
    inversion_pairs = {}

    card1_pickrate = 0
    card2_pickrate = 0

    # Get all pairs and their pick rates
    for pair in pairs.keys():
        card1, card2 = get_cards_from_pair(pair, card_data)
        card1_name = card1.name
        card2_name = card2.name

        if debug:
            print(f"Computing inversion for {card1_name} and {card2_name}")

        card1_pickrate = pairs[pair][0]
        card2_pickrate = pairs[pair][1]

        if debug:
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

        #inversion = winrate_difference * pickrate_difference
        inversion = pickrate_difference

        more_picked_card_name = more_picked_card.name
        less_picked_card_name = higher_winrate_card.name

        # Store the pair and the pick rates
        inversion_pairs[(more_picked_card_name, less_picked_card_name)] = {"higher_pickrate": higher_pickrate,
            "lower_pickrate": lower_pickrate,
            "higher_winrate": higher_winrate,
            "lower_winrate": lower_winrate,
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
    card1Subs, card1Comps = findTopSubstitutes(card1, cards)
    card2Subs, card2Comps = findTopSubstitutes(card2, cards)

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


def check_if_substitutes(results, numControls, labels):

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


# for picks where poth cards were present
# compute the rate at which card 1 was picked
# and the rate at which card 2 was picked
# should sum to 1
def compute_pairwise_pickrates(pairs, drafts):
    print("Computing pairwise pickrates")

    # If there is a local cache, use it
    if os.path.exists(os.path.join(os.path.dirname(__file__),
                                    "..",
                                    "data",
                                    "pairwise_pickrates.pickle")):
        print("Found local cache of pairwise pickrates")
        with open(os.path.join(os.path.dirname(__file__),
                               "..",
                               "data",
                               "pairwise_pickrates.pickle"),
                  "rb") as f:
            return pickle.load(f)

    pairs_with_pickrates = {}
    # Compute the pairwise pick rate for each pair
    for card1, card2 in pairs:
        pickrate = compute_pairwise_pickrate(drafts, card1, card2)
        pairs_with_pickrates[card1, card2] = pickrate

    # Save the results to a local cache
    with open(os.path.join(os.path.dirname(__file__),
                           "..",
                           "data",
                           "pairwise_pickrates.pickle"),
              "wb") as f:
        pickle.dump(pairs_with_pickrates, f)

    return pairs_with_pickrates


# Filter by number of some key card in the pool
def elasticity_substitution(card1,
                            subs_list,
                            check_card1_colour=True,
                            check_card2_colour=False,
                            pick_number=True):
    if debug:
        print(f"Checking substitution for {card1}")
        print(f"Checking cards: {subs_list}")

    # Allow function to take a single substitute
    if type(subs_list) is str:
        subs_list = [subs_list]

    # Keep a running total for indexes
    num_controls = 0
    # Constant term
    num_controls += 1

    # Pick number
    if pick_number:
        num_controls += 1

    # Theoden in the pool
    #num_controls += 1

    # number of card1 in pool
    num_controls += 1

    card1_obj = name_to_card(card1, card_data)
    card1_colour = card1_obj.colour

    card2_obj = None
    card2_colour = None

    if check_card1_colour:
        num_controls += 1

    if check_card2_colour:
        card2_obj = name_to_card(subs_list[0], card_data)
        card2_colour = card2_obj.colour

        if card1_colour == card2_colour:
            check_card2_colour = False

            if debug:
                print(f"Card 1 and card 2 are the same colour: {card1_colour}")
                print("Will not check for card 2 colour")
        else:
            num_controls += 1

    y = []
    card1_colour_count = []
    card1_in_pool = []

    card2_colour_count = []
    num_substitutes = {}

    for i in range(NUM_MULTIPLES + 1):
        num_substitutes[i] = []

    # If card1 is passed in the list of substitutes, remove it
    # This is because we *always* use number of card1 already in pool
    # as a control
    if card1 in subs_list:
        subs_list.remove(card1)

    for pick in card1_obj.picks:
        pick_name = cardNamesHash[pick.pick]

        if pick_name == card1:
            y.append(1)
        else:
            y.append(0)

        # Append number of cards that are the same colour as card1
        if check_card1_colour:
            card1_colour_total = 0
            for colour_of_card in pick.colourInPool.keys():
                if card1_colour in colour_of_card:
                    card1_colour_total += pick.colourInPool[colour_of_card]

            # Take the log of the number of cards
            # This is to account for diminishing returns
            card1_colour_total = math.log(card1_colour_total + 1)
            card1_colour_count.append(card1_colour_total)

        if check_card2_colour:
            card2_colour_total = 0
            for colour_of_card in pick.colourInPool.keys():
                if card2_colour in colour_of_card:
                    card2_colour_total += pick.colourInPool[colour_of_card]
            # Take the log of the number of cards
            # This is to account for diminishing returns
            card2_colour_total = math.log(card2_colour_total + 1)
            card2_colour_count.append(card2_colour_total)

        substitutes_count = 0
        # Count total number of substitutes in the pool
        for card_name in subs_list:
            card_num = cardNumsHash[card_name]
            if card_num in pick.numCardInPool:
                substitutes_count += pick.numCardInPool[card_num]
                break

        for i in num_substitutes.keys():
            if substitutes_count == i:
                num_substitutes[i].append(1)
            else:
                num_substitutes[i].append(0)

        # Control for the number of card1 in the pool
        card1_num = cardNumsHash[card1]
        if card1_num in pick.numCardInPool:
            card1_in_pool.append(pick.numCardInPool[card1_num])
        else:
            card1_in_pool.append(0)

    # Eliminate values that didn't have enough observations
    temp = {}
    num_observations = []
    for i in range(1, NUM_MULTIPLES + 1):
        total_observations = sum(num_substitutes[i])
        num_observations.append(total_observations)

        if total_observations < OBSERVATIONS_THRESHOLD:
            if debug:
                print(f"Not enough observations with {i} cards in the pool.")
                print("Will eliminate all higher values")
            break
        else:
            temp[i] = num_substitutes[i]

    num_substitutes = temp

    endog = np.array(y)

    # Assemble exog matrix
    exog = None

    if pick_number:
        # Create list of all pick numbers
        pick_numbers = []
        for pick in card1_obj.picks:
            pick_numbers.append(pick.pick_number)

        # Add the pick number
        exog = np.array(pick_numbers)

    # Create the array that is 1 if there's a Theoden in pool and 0 otherwise
    #theoden_in_pool = []
    #theoden_number = cardNumsHash["Rally at the Hornburg"]
    #for pick in card1_obj.picks:
    ##    if pick.numCardInPool[theoden_number] > 0:
     ##       theoden_in_pool.append(1)
       #     if debug:
        #        print(f"Found Theoden in the pool at pick {pick.pick_number}")
        #else:
        #    theoden_in_pool.append(0)

    #exog = np.array(theoden_in_pool)

    if check_card1_colour:
        exog = np.array(card1_colour_count)
        #exog = np.column_stack((exog, card1_colour_count))

        if check_card2_colour:
            exog = np.column_stack((exog, card2_colour_count))

        exog = np.column_stack((exog, card1_in_pool))
    else:
        exog = np.array(card1_in_pool)

    # Add the number of substitutes in the pool
    for i in num_substitutes.keys():
        exog = np.column_stack((exog, num_substitutes[i]))

    # Add a constant term
    exog = sm.add_constant(exog)

    model = sm.OLS(endog, exog)

    # Label the variables in the model
    labels = ["Constant"]

    if pick_number:
        labels.append("Pick number")

    #labels.append("Theoden in pool")

    if check_card1_colour:
        labels.append(f"{card1_colour} cards in pool")
    if check_card2_colour:
        labels.append(f"{card2_colour} cards in pool")

    labels.append(f"{card1} in pool")

    if len(subs_list) == 1:
        for i in num_substitutes.keys():
            labels.append(f"{i} {subs_list[0]} in pool")
    else:
        for i in num_substitutes.keys():
            labels.append(f"{i} substitutes in pool")

    model.exog_names[:] = labels

    results = model.fit()

    # append total observations at each number to the results
    # We use this to calculate our elasticity of substitution
    results.nobs = num_observations

    substitutes, elasticity = check_if_substitutes(results,
                                                   num_controls,
                                                   labels)

    if debug:
        if substitutes:
            print(f"{card1} and {subs_list} are substitutes")
        else:
            print(f"{card1} and {subs_list} are not substitutes")

    print(f"Elasticity of substitution for {card1} and {subs_list}: {elasticity}")

    return elasticity


# Regress all cards ALSA on the number of subsitutes seen in the sample
# and on their GIH winrate
# Takes a list of cards
def regress_alsa(cards):
    print("Regressing ALSA on number of substitutes and GIH winrate")

    # Make a data frame for the regression
    # Each row is a card
    # Each column is a variable
    # The variables are:
    # 1. Card name
    # 2. Number of substitutes seen
    # 3. GIH winrate
    # 4. Pickrate

    regression_data = []

    # Populate the data frame
    for card in cards:
        card_name = card.name
        try:
            number_substitutes = card.numberSubstitutes
        except AttributeError as e:
            print(f"Card {card_name} does not have a number of substitutes")
            print(e)
            exit(1)
        gih_winrate = card.gameInHandWinrate
        alsa = card.alsa

        regression_data.append([card_name,
                                number_substitutes,
                                gih_winrate,
                                alsa])

    # Sort on number of substitutes
    regression_data.sort(key=lambda x: x[1])

    # Print all
    for row in regression_data:
        print(f"{row[0]}: number_substitutes: {row[1]}, gih_winrate: {row[2]}, alsa: {row[3]}")

    # Create the data frame
    df = pd.DataFrame(regression_data, columns=["card_name",
                                                "number_substitutes",
                                                "gih_winrate",
                                                "alsa"])

    # Run the regression
    # The model is:
    # pickrate = b0 + b1 * number_substitutes + b2 * gih_winrate
    # The null hypothesis is that b1 = 0
    # The alternative hypothesis is that b1 < 0
    # This is a one-sided test

    # Create the model
    model = sm.OLS.from_formula("alsa ~ number_substitutes + gih_winrate",
                                data=df)

    # Fit the model
    results = model.fit()

    # Print the results
    print(results.summary())

    # Graph the results
    # Plot the residuals
    fig = plt.figure(figsize=(12, 8))
    fig = sm.graphics.plot_regress_exog(results, "number_substitutes", fig=fig)

    # Show the graph
    plt.show()

    # Plot the residuals
    fig = plt.figure(figsize=(12, 8))
    fig = sm.graphics.plot_regress_exog(results, "gih_winrate", fig=fig)

    # Show the graph
    plt.show()


# pairs: list of tuples (card1 & card2, elasticity)
def get_substitutes(cards):
    print("Getting substitutes for a list of cards")

    # Check if there is a local cache
    if os.path.exists(os.path.join(os.path.dirname(__file__), "..", "data", "substitutes.pickle")):
        with open(os.path.join(os.path.dirname(__file__), "..", "data", "substitutes.pickle"), "rb") as f:
            return pickle.load(f)

    cards_with_subs = {}

    cards_with_elasticities = {}

    # Compute the elasticity of substitution for each pair of cards
    for card1 in cards:
        for card2 in cards:
            if card1 != card2:
                elasticity = elasticity_substitution(card1,
                                                     card2,
                                                     check_card1_colour=True,
                                                     check_card2_colour=False,
                                                     pick_number=True)
                cards_with_elasticities[card1, card2] = elasticity

    # Two cards are substitutes if the elasticiy of substitution card1, card2 < 0 
    # Add the substitutes and their elasticities to the dictionary
    for card1 in cards:
        cards_with_subs[card1] = []
        for card2 in cards:
            if card1 != card2:
                elasticity1 = cards_with_elasticities[card1, card2]
                elasticity2 = cards_with_elasticities[card2, card1]
                if elasticity1 < 0:
                    # Using elasticity1 here means that we are weighting
                    # by the amount that having card2 in the pool
                    # decreases the pick rate of card1
                    # This correlates with card quality
                    cards_with_subs[card1].append((card2, elasticity1))
                    # Whereas using elasticity2 here means that we are weighting
                    # by the amount that having card1 in the pool
                    # decreases the pick rate of card2
                    #cards_with_subs[card1].append((card2, elasticity2))

    # Cache the cards with their substitutes
    with open(os.path.join(os.path.dirname(__file__), "..", "data", "substitutes.pickle"), "wb") as f:
        pickle.dump(cards_with_subs, f)
        # delete availabilities.pickle
        # if it exists
        # Because it is dependent on the substitutes
        if os.path.exists(os.path.join(os.path.dirname(__file__), "..", "data", "availabilities.pickle")):
            os.remove(os.path.join(os.path.dirname(__file__), "..", "data", "availabilities.pickle"))

    return cards_with_subs


# For each inversion pair, calculate the difference in substitutes seen
# Also calculate the difference in pick rate between the two cards
# Regress the difference in pick rate on the difference in substitutes seen
# Create a list of tuples
# Each tuple is a pair of cards
# The first card is the more picked card
# The second card is the less picked card
# The third element is the difference in pick rate between the two cards
# The fourth element is the difference in substitutes seen between the cards
def regress_inversion_pairs(inversionPairs, card_data):
    print("Regressing pick rate difference on substitutes seen difference")

    dependent_var = "Pickrate Difference (Card 1 - Card 2)"
    independent_var = "Substitutes Seen Difference (Card 1 - Card 2)"

    regr_data = []

    for pair in inversionPairs.keys():
        card1, card2 = get_cards_from_pair(pair, card_data)

        if type(card1) is Card:
            card1 = card1.name
        if type(card2) is Card:
            card2 = card2.name

        # Get the coefficient of inversion
        inversion = inversionPairs[pair]["inversion"]

        # Count the total number of observations of the substitutes
        card1_subs_count = card_data[card1].numberSubstitutes
        card2_subs_count = card_data[card2].numberSubstitutes

        # At this point, card1 is the more picked card
        # because of how find_inversion_pairs works

        # We want the difference between substitutes seen for the more picked card
        # and the less picked card
        subs_count_diff = card1_subs_count - card2_subs_count

        # Skip pairs where the coefficient of inversion is 0
        # These are pairs where players are indifferent between the two cards
        if inversion == 0:
            continue

        # Append the tuple to the list
        regr_data.append((card1,
                          card2,
                          inversion,
                          subs_count_diff))

    # Create a dataframe from the regression data
    df = pd.DataFrame(regr_data, columns=["Card 1",
                                               "Card 2",
                                               dependent_var,
                                               independent_var])

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


def compute_availability(cards_with_subs, card_data):
    print("Computing availability of substitutes")

    # Check if there is a local cache
    # availiabilities.pickle
    # if so return it
    if os.path.exists(os.path.join(os.path.dirname(__file__), "..", "data", "availabilities.pickle")):
        with open(os.path.join(os.path.dirname(__file__), "..", "data", "availabilities.pickle"), "rb") as f:
            return pickle.load(f)

    availabilities = {}

    for card, substitutes in cards_with_subs.items():
        # Go through each card and compute the number of substitutes seen
        # This is the total number of observations of the substitutes
        # Calculate the number of observations of the substitutes
        # Add to the card object
        cardSubstitutesCount = 0
        for card2, card2_elas in substitutes:
            # Weight the number of observations by the elasticity
            # This is to account for the fact that some substitutes
            # are closer than others
            sub_availability = abs(card_data[card2].numberSeen * card2_elas / num_drafts)
            cardSubstitutesCount += sub_availability

            #cardSubstitutesCount += 1

        availabilities[card] = cardSubstitutesCount

    # Cache the availabilities
    with open(os.path.join(os.path.dirname(__file__), "..", "data", "availabilities.pickle"), "wb") as f:
        pickle.dump(availabilities, f)

    return availabilities


def get_cards_in_colours(card_data, colours):
    cards = []

    all_colours = ["W", "U", "B", "R", "G"]

    other_colours = []
    for colour in all_colours:
        if colour not in colours:
            other_colours.append(colour)

    for card in card_data.values():

        # Exclude mythics and rares
        if card.rarity == "R" or card.rarity == "M":
            continue

        # Cards must contain at least one of the colours
        if not any(colour in card.colour for colour in colours):
            continue

        # Cards must not contain any of the other colours
        if any(colour in card.colour for colour in other_colours):
            continue

        cards.append(card.name)

    return cards

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

GET_DRAFTS_FROM_CACHE = os.path.exists(os.path.join(os.path.dirname(__file__), "..", "data", "drafts.pickle"))

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

cards = []

cards = get_cards_in_colours(card_data, ["R", "B"])

pairs = []
for card1 in cards:
    for card2 in cards:
        if card1 != card2:
            pairs.append((card1, card2))

# Compute the number of pairs
numPairs = len(pairs)
print(f"Number of pairs: {numPairs}")

# Eliminate drafs where the drafter didn't end up with at least 6 black cards
# And at least 6 red cards
# This is to eliminate drafts where the drafter was not in the right colours

for draft in drafts:
    numRed = 0
    numBlack = 0
    for pick in draft.picks:
        if pick.pick in cards:
            if card_data[pick.pick].colour.contains("R"):
                numRed += 1
            elif card_data[pick.pick].colour.contains("B"):
                numBlack += 1

    if numRed < 7 or numBlack < 7:
        drafts.remove(draft)

# Print the number of remaining drafts
print(f"Number of drafts after eliminating drafts where drafter did not end up with at least 6 red and 6 black cards: {len(drafts)}")

print("=====================================")

# Overwrite the cache to eliminate drafts where the drafter did not end up with at least 6 red and 6 black cards
with open(os.path.join(os.path.dirname(__file__), "..", "data", "drafts.pickle"), "wb") as f:
    pickle.dump(drafts, f)

timestamp = time.time()
parsePoolInfo(cards, drafts)
print(f"Time to parse pool info: {time.time() - timestamp}")

# For each pair, when each card is on offer, compute the pick rate of each card
# If there is a local cache, read from that
# Otherwise, compute the pairwise pick rates

pairs = compute_pairwise_pickrates(pairs, drafts)

# For each red card, compute its substitutes
# This is all the cards with elasticity of substitution < -0.005
# This is a list of tuples
# The first element is the card name
# The second element is the elasticity of substitution
# The third element is the number of observations of substitutes

cards_with_subs = get_substitutes(cards)

# Compute the number of substitutes seen for each card
availiabilities = compute_availability(cards_with_subs, card_data)

# Print the cards sorted by their availability
availiabilities = {k: v for k, v in sorted(availiabilities.items(),
                                           key=lambda item: item[1],
                                           reverse=True)}

print("All cards along with their substitutes")
print("=====================================")
for card in availiabilities.keys():
    print(f"{card}: {availiabilities[card]}")
    # Print its substitutes
    substitutes = cards_with_subs[card]
    for substitute in substitutes:
        print(f"{substitute[0]}: {substitute[1]}")

    print("=====================================")


# Add to card_data as a field of the card object
for card in card_data.values():
    if card.name in availiabilities.keys():
        card.numberSubstitutes = availiabilities[card.name]
    else:
        card.numberSubstitutes = 0

# Find pairs where the card with the lower GIH winrate is picked more often
# If there is a local cache, read from that
# Otherwise, compute the inversion pairs
inversionPairs = {}
if os.path.exists(os.path.join(os.path.dirname(__file__), "..", "data", "inversionPairs.pickle")):
    with open(os.path.join(os.path.dirname(__file__), "..", "data", "inversionPairs.pickle"), "rb") as f:
        inversionPairs = pickle.load(f)
else:
    inversionPairs = find_inversion_pairs(pairs, card_data, only_return_inverted=True)

    # Cache the inversion pairs
    with open(os.path.join(os.path.dirname(__file__), "..", "data", "inversionPairs.pickle"), "wb") as f:
        pickle.dump(inversionPairs, f)

# Sort availabilities by value
# Ascending
availiabilities = {k: v for k, v in sorted(availiabilities.items(),
                                             key=lambda item: item[1],
                                             reverse=False)}

cards = list(availiabilities.keys())

card_objs = []
for card in cards:
    card_objs.append(card_data[card])

regress_alsa(card_objs)

regress_inversion_pairs(inversionPairs, card_data)

# Print the time that script took to run
print("Time to run: " + str(time.time() - timestamp))
