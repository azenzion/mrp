# Import the draft data
import itertools
import os
import csv
import time
import pickle
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

NUM_DRAFTS = 162153
#NUM_DRAFTS = 10000

# The number of multiples in the pool to consider
# This just needs to be larger than we're likely to see
# in the data
# In practice this is like, 4 or 5 at most
NUM_MULTIPLES = 10

# For a number x
# If there are less drafts than this with x of card2 in pool
# we will not consider it
OBSERVATIONS_THRESHOLD = 40

# If the computed elasticity of substitution is greater than this
# the cards are not substitutes
ELASTICITY_THRESHOLD = -0.01

COLOURS_THRESHOLD = 5

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

    # We want all the colour combinations
    # This is a list of all the possible combinations
    # of the colours
    # We use this to get the colour pairs
    # and the colour triples
    colour_combinations = []
    for i in range(1, len(COLOURS) + 1):
        colour_combinations.extend(itertools.combinations(COLOURS, i))

    # Convert each tuple to a string
    colour_combinations = ["".join(colour) for colour in colour_combinations]

    return colour_combinations


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


def name_to_colour(card_name, card_data):

    # If given a number, convert to a string
    if type(card_name) is int:
        card_name = cardNamesHash[card_name]

    if card_name in card_data:
        return card_data[card_name].colour

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


def parse_drafts(csv_file_path, ltr_cards, numDrafts):
    drafts = []

    DRAFTS_CACHE = os.path.join(os.path.dirname(__file__),
                                '..',
                                'data',
                                'drafts.pickle')

    # If there is a local cache, use it
    if os.path.exists(DRAFTS_CACHE):
        print("Found local cache of drafts")
        with open(DRAFTS_CACHE, "rb") as f:
            drafts = pickle.load(f)
        # If the length of drafts is not within
        # one order of magnitude of numDrafts
        # delete the local cache and proceed with reading csv
        if len(drafts) < numDrafts / 10 or len(drafts) > 10 * numDrafts:
            print("Length of drafts does not match numDrafts")
            print(f"Length of drafts: {len(drafts)}")
            print(f"numDrafts: {numDrafts}")
            print("Deleting all local caches and proceeding with reading csv")

            # delete all ".pickle" files in the data directory
            #for filename in os.listdir(os.path.join(os.path.dirname(__file__), "..", "data")):
            #    if filename.endswith(".pickle"):
            #        os.remove(os.path.join(os.path.dirname(__file__), "..", "data", filename))

            # Proceed with reading csv
            drafts = []
        else:
            return drafts

    csv_reader = None
    print('begin parsing draft data')
    with open(csv_file_path, "r") as f:
        # Create a CSV reader
        csv_reader = csv.reader(f, delimiter=",")

        # Read the header row
        header_row = next(csv_reader)
        if debug:
            print(header_row)

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
            pick.pack_number = int(row[7])
            pick.pick_number = int(row[8])
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

    # Save the parsed results to a local cache
    with open(os.path.join(os.path.dirname(__file__),
                           "..",
                           "data",
                           "drafts.pickle"),
              "wb") as f:
        pickle.dump(drafts, f)

    print('Found ' + str(len(drafts)) + ' drafts')

    return drafts


# Takes a list of pairs with their pairwise pick rates
# Returns a dictionary
# Keys are pairs
# More Picked Card, Less Picked Card
# Values are higher pick rate, lower pick rate, higher winrate, lower winrate, inversion
def find_inversion_pairs(pairs,
                         card_data,
                         only_return_inverted=True,
                         simple_inversion_score=False,
                         ) -> dict:
    print("Finding inversion pairs")

    # If there is a local cache, use it
    if os.path.exists(os.path.join(os.path.dirname(__file__),
                                    "..",
                                    "data",
                                    "inversion_pairs.pickle")):
        print("Found local cache of inversion pairs")
        with open(os.path.join(os.path.dirname(__file__),
                                 "..",
                                 "data",
                                 "inversion_pairs.pickle"),
                    "rb") as f:
                return pickle.load(f)

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

        if simple_inversion_score:
            inversion = winrate_difference
        else:
            inversion = winrate_difference * pickrate_difference

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

    # Save the results to a local cache
    with open(os.path.join(os.path.dirname(__file__),
                           "..",
                           "data",
                           "inversion_pairs.pickle"),
              "wb") as f:
        pickle.dump(inversion_pairs, f)

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


def append_regr_info_to_card(card, pick):
    # The information we need to append to the card is
    # 1 was the card picked
    # 2 how many copies of the card were in the pool
    # 3 how many copies of the card were in the pool of the same colour
    # num_cards in pool to calculate subsitutes later

    pick_for_regr = {}
    pick_for_regr["wasPicked"] = 0
    pick_for_regr["sameCardInPool"] = 0
    pick_for_regr["sameColourInPool"] = 0
    pick_for_regr["numCardInPool"] = {}
    pick_for_regr["pack_number"] = pick.pack_number

    # Check if the card was picked
    if pick.pick == cardNumsHash[card.name]:
        pick_for_regr["wasPicked"] = 1

    # Check if the card was in the pool
    if card.name in pick.numCardInPool:
        pick_for_regr["sameCardInPool"] = pick.numCardInPool[card.name]

    # Check if the card was in the pool of the same colour
    if card.colour in pick.colourInPool:
        pick_for_regr["sameColourInPool"] = pick.colourInPool[card.colour]

    pick_for_regr["numCardInPool"] = pick.numCardInPool

    pick_for_regr["pick_number"] = pick.pick_number

    card.picks.append(pick_for_regr)



def dict_increment(dictionary, key):
    if key in dictionary:
        dictionary[key] += 1
    else:
        dictionary[key] = 1


def process_draft_pool(draft):

    # initialize values
    num_card_in_draft_pool = {}
    colour_in_draft_pool = {}

    for pick in draft.picks:

        # Convert from number to name
        pick_name = cardNamesHash[pick.pick]

        # Log card
        dict_increment(num_card_in_draft_pool, pick_name)

        pool_card_colour = name_to_colour(pick_name, card_data)
        # Log colour of card
        dict_increment(colour_in_draft_pool, pool_card_colour)

        # Add the pool cards to pick.numCardInPool
        pick.numCardInPool = num_card_in_draft_pool
        pick.colourInPool = colour_in_draft_pool

        # Pool analysis
        # Count number of coloured cards in the pool
        for pool_card in num_card_in_draft_pool.keys():
            card = card_data[pool_card]

        append_regr_info_to_card(card, pick)


# This sets us up for the regression
# And lets us only store information about those cards
# in the pool which we care about
# This is a memory optimization
# We go trough each draft
# For each pick, we go through the pool
# If the card is in the pool, we count it
def parse_pool_info(drafts):
    timestamp = time.time()
    if debug:
        print(f"Computing draft stats for {len(drafts)} drafts")
    else:
        print(f"Computing draft stats")

    # Make a dictionary
    # Keys are pairs of cards
    # Values is a list of ints
    # where a 0 indicates that card1 was chosen
    # and a 1 indicates that card2 was chosen
    # We use this for pairwise pick rates

    for draft in drafts:
        process_draft_pool(draft)

    print(f"Time to parse pool info: {time.time() - timestamp}")

    return card_data


def compareSubstitutes(card1, card2, cardList, card_data):

    # Compute the substitutes for a card within a list of cards
    # hang on to the complements even though we don't use them at the moment
    card1Subs, card1Comps = findTopSubstitutes(card1, cards)
    card2Subs, card2Comps = findTopSubstitutes(card2, cards)

    # print the total number of substitutes for each
    print(f"Number of substitutes for card1: {len(card1Subs)}")
    print(f"Number of substitutes for card2: {len(card2Subs)}")

    # card1Subs = [x for x in rallySubs if x[1] < -0.005]
    # card2Subs = [x for x in smiteSubs if x[1] < -0.005]

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


# num_obs is a dictionary where keys i are number of card2
def check_if_substitutes(results,
                         num_controls,
                         num_obs,
                         weight_by_num_obs,
                         eliminate_zero_coef,
                         labels):

    coeff = 0
    for i in range(num_controls - 1, len(results.params)):
        if debug:
            print(f'coefficient {labels[i]}: {results.params[i]}')
        coeff += results.params[i] * num_obs[i]

    return coeff < 0, coeff

    # Are we or are we not checking the 0th coefficient
    if eliminate_zero_coef:
        first_coef = num_controls
    else:
        first_coef = num_controls - 1

    elasticity = 0.0

    # Total number of observations
    total_obs = 0

    # Get the coefficent of each number of substitutes in the pool
    # Calculate the difference between having i substitutes and i + 1 substitutes
    STARTING_INDEX = 0

    num_subs = STARTING_INDEX

    for i in range(first_coef, len(results.params) - 1):

        # Get the coefficient for having i + 1 substitutes
        coefficient_greater = results.params[i + 1]
        # Get the coefficient for having i substitutes
        coefficient_less = results.params[i]

        # If we're weighting by number of observations
        # Get the number of observations for each number of substitutes
        num_obs_greater = num_obs[num_subs + 1]

        if weight_by_num_obs:
            elasticity += (coefficient_greater - coefficient_less) * num_obs_greater * (i+1)
        else:
            elasticity += (coefficient_greater - coefficient_less)

        total_obs += num_obs_greater

        num_subs += 1

    if total_obs == 0:
        if debug:
            print('Not enough observations to calculate elasticity of substitution')
        return False, 999

    # divide by totalNobs
    if weight_by_num_obs:
        elasticity /= total_obs

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


def count_substitutes(pick,
                      subs_list,
                      num_substitutes,
                      card1):
    substitutes_count = 0

    # Count total number of substitutes in the pool
    for card_name in subs_list:
        if card_name in pick['numCardInPool']:
            substitutes_count += pick['numCardInPool'][card_name]

    for i in num_substitutes.keys():
        if substitutes_count == i:
            num_substitutes[i].append(1)
        else:
            num_substitutes[i].append(0)

    return num_substitutes


def create_exog(num_subs,
                check_card1_in_pool,
                card1_in_pool,
                check_card1_colour,
                card1_colour_count,
                check_card2_colour,
                card2_colour_count,
                pick_number,
                pick_number_list,
                subs_list,
                card1_colour,
                card2_colour,
                card1_name,
                ):

    # Constant term
    exog = np.ones((len(card1_in_pool), 1))
    labels = ["Constant"]

    if pick_number:
        exog = np.column_stack((exog, pick_number_list))
        labels.append("Pick number")

    if check_card1_colour:
        exog = np.column_stack((exog, card1_colour_count))
        labels.append(f"{card1_colour} in pool (log)")

    if check_card2_colour:
        exog = np.column_stack((exog, card2_colour_count))
        labels.append(f"{card2_colour} in pool(log)")

    if check_card1_in_pool:
        exog = np.column_stack((exog, card1_in_pool))
        labels.append(f"{card1_name} in pool")

    # Theoden in the pool
    #labels.append("Theoden in pool")

    # Add the number of substitutes in the pool
    for i in num_subs.keys():
        exog = np.column_stack((exog, num_subs[i]))

        # If there is only one substitute, label it as such
        if len(subs_list) == 1:
            labels.append(f"{i} {subs_list[0]} in pool")
        else:
            labels.append(f"{i} substitutes in pool")

    # At this point, labels and results.params are the same length
    # If not
    # Print the labels and the params
    if len(labels) != len(exog[0]):
        print("ERROR: Length of labels and params do not match")
        print(f"Length of labels: {len(labels)}")
        print(f"Length of exog: {len(exog[0])}")
        print(labels)
        print(exog)
        exit(1)

    return exog, labels


def eliminate_low_observations(num_substitutes):
    temp = {}
    num_observations = {}

    for i in range(1, NUM_MULTIPLES):
        total_observations = sum(num_substitutes[i])

        if debug:
            print(f"Number of observations with {i} substitutes: {total_observations}")

        num_observations[i] = total_observations

        if total_observations < OBSERVATIONS_THRESHOLD:
            if debug:
                print(f"Not enough observations with {i} cards in the pool.\
                       Will eliminate all higher values")
            break
        else:
            temp[i] = num_substitutes[i]

    num_substitutes = temp

    return num_substitutes, num_observations


# Count the number of cards in pick.colourInPool
# that are the same colour as card
# excluding cards in excludes
def get_colour_in_pool(pick,
                        card,
                        excludes):
    card_colour_in_pool = 0
    card_colour = name_to_colour(card, card_data)

    for pool_card_name in pick['numCardInPool'].keys():
        pool_card_colour = name_to_colour(pool_card_name, card_data)
        if pool_card_colour == card_colour and pool_card_name not in excludes:
            card_colour_in_pool += pick['numCardInPool'][pool_card_name]

    return card_colour_in_pool


# Filter by number of some key card in the pool
def elasticity_substitution(card1,
                            subs_list,
                            check_card1_colour,
                            check_card2_colour,
                            pick_number,
                            check_card1_in_pool,
                            ):

    # If card1 is passed in the list of substitutes, remove it
    # This is because we *always* use number of card1 already in pool
    # as a control
    if check_card1_in_pool and card1 in subs_list:
        subs_list.remove(card1)

    # Allow function to take a single substitute
    if type(subs_list) is str:
        subs_list = [subs_list]

    # If card1 and some card in subs_list are different colours
    # check_card1_colour and check_card2_colour will both be true
    card1_obj = name_to_card(card1, card_data)
    card1_colour = card1_obj.colour

    card2 = subs_list[0]
    card2_obj = name_to_card(subs_list[0], card_data)
    card2_colour = card2_obj.colour

    # If card1 and card2 are not the same colour
    # check card1 colour and card2 colour

    # This accounts for the fact that an R card
    # and a BR card are both red
    if card1_colour not in card2_colour and\
       card2_colour not in card1_colour:
        pass
        #check_card1_colour = True
    #    check_card2_colour = True
       # debug = True
        #pick_number = True
        #check_card1_colour = True
        #check_card2_colour = True

    # Keep a running total for indices
    num_controls = 0

    # Constant term
    num_controls += 1

    # Pick number
    if pick_number:
        num_controls += 1

    # number of card1 in pool
    num_controls += 1

    if check_card1_colour:
        num_controls += 1

    if check_card2_colour:
        num_controls += 1

    if check_card1_in_pool:
        num_controls += 1

    # This is our dependent variable
    # 1 if card1 was picked, 0 otherwise
    picked = []

    # This is our exogenous variable
    # The number of cards of the same colour as card1 in the pool
    card1_colour_count, card2_colour_count = [], []

    # Number of card1 in the pool
    # This is to control for the self-substitution
    # or self-complementarity
    # Diminishing or increasing returns to having more of the same card
    card1_in_pool = []

    # Number of substitutes in the pool
    # Keys are number of substitutes
    # Values are 1 if there were that many substitutes in the pool 
    # for this pick
    num_substitutes = {}

    for i in range(1, NUM_MULTIPLES + 1):
        num_substitutes[i] = []

    simple_counts = []

    # picked is just concatentating pick.wasPicked for each pick
    for pick in card1_obj.picks:
        picked.append(pick['wasPicked'])

        # Count the number of cards that are the same colour as card1 in the pool
        if check_card1_colour:
            card1_colour_in_pool = get_colour_in_pool(pick,
                                                      card1,
                                                      [card1, card2])
            # Take the log due to diminishing returns
            card1_colour_in_pool = np.log(card1_colour_in_pool + 1)

            card1_colour_count.append(card1_colour_in_pool)

        # Count the number of cards the same colour as card2 in the pool
        if check_card2_colour:
            card2_colour_in_pool = get_colour_in_pool(pick,
                                                      card2,
                                                      [card1, card2])

            # Take the log due to diminishing returns
            card2_colour_in_pool = np.log(card2_colour_in_pool + 1)

            card2_colour_count.append(card2_colour_in_pool)

        card1_in_pool.append(pick['sameCardInPool'])

        num_substitutes = count_substitutes(pick,
                                            subs_list,
                                            num_substitutes,
                                            card1)

        if card2 in pick['numCardInPool']:
            simple_count = pick['numCardInPool'][card2]
        else:
            simple_count = 0
        simple_counts.append(simple_count)

        if False:
            print(f"Number of {card1} in pool: {pick['sameCardInPool']}")
            if card2 in pick['numCardInPool']:
                print(f"Number of {card2} in pool: {pick['numCardInPool'][card2]}")
            else:
                print(f"Number of {card2} in pool: 0")
            print(f"Number of {card1_colour} cards in pool: {card1_colour_in_pool}")
            print(f"Number of {card2_colour} cards in pool: {card2_colour_in_pool}")

    # Eliminate values that didn't have enough observations
    num_substitutes, num_observations = eliminate_low_observations(num_substitutes)

    if debug:
        for i in range(1, len(num_substitutes.keys())):
            print(f"Number of times {card1} was on offer with {i} {card2} in pool")
            print(sum(num_substitutes[i]))

    # Create the pick_number array
    pick_number_list = []

    if pick_number:
        for pick in card1_obj.picks:
            # Since there are 3 packs (0, 1, 2), need to append 15 * pack_number
            # to get the pick number
            pick_number = pick['pick_number'] + 1
            pack_number = pick['pack_number']
            modified_pick_number = 15 * pack_number + pick_number
            modified_pick_number = np.log(modified_pick_number)
            pick_number_list.append(modified_pick_number)

    # Create the matrices for the regression
    endog = np.array(picked)

    exog, labels = create_exog(num_substitutes,
                               check_card1_in_pool,
                               card1_in_pool,
                               check_card1_colour,
                               card1_colour_count,
                               check_card2_colour,
                               card2_colour_count,
                               pick_number,
                               pick_number_list,
                               subs_list,
                               card1_colour,
                               card2_colour,
                               card1)

    # Create simple exog, for a regression of picked
    # on number of card2 in pool

    simple_exog = np.array(simple_counts)

    # prepend a column of ones for the constant
    simple_exog = np.column_stack((np.ones(len(simple_exog)), simple_exog))

    simple_labels = ["Constant", f"{card2} in pool"]

    # Create simple model
    simple_model = sm.OLS(endog, simple_exog)

    simple_model.exog_names[:] = simple_labels

    simple_results = simple_model.fit()

    #print(simple_results.summary())
   # exit()
    
    


    # Create the model
    model = sm.OLS(endog, exog)

    model.exog_names[:] = labels

    results = model.fit()

    substitutes, elasticity = check_if_substitutes(results,
                                                   num_controls,
                                                   num_observations,
                                                   weight_by_num_obs=True,
                                                   eliminate_zero_coef=True,
                                                   labels=labels)

    if debug:
        print(results.summary())
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
        card_obj = name_to_card(card, card_data)
        try:
            number_substitutes = card_obj.numberSubstitutes
        except AttributeError as e:
            print(f"Card {card} does not have a number of substitutes")
            print(e)
            exit(1)
        gih_winrate = card_obj.gameInHandWinrate
        alsa = card_obj.alsa

        regression_data.append([card,
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


# Get all the substitutes for a list of cards
# Takes a list of cards
# Returns {
#  card1: [(sub1, elasticity1), (sub2, elasticity2), ...],
#  card2: ...
# }
def get_substitutes(cards,
                    symmetrical_subs) -> dict:
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
                                                     check_card1_colour=False,
                                                     check_card2_colour=False,
                                                     pick_number=False,
                                                     check_card1_in_pool=True,
                                                     )
                cards_with_elasticities[card1, card2] = elasticity

    # Two cards are substitutes if the elasticiy of substitution card1, card2 < 0
    # They must also have comparable game in hand winrates
    # Add the substitutes and their elasticities to the dictionary
    for card1 in cards:
        cards_with_subs[card1] = []
        for card2 in cards:
            if card1 != card2:
                elasticity1 = cards_with_elasticities[card1, card2]
                elasticity2 = cards_with_elasticities[card2, card1]
                if elasticity1 >= 0:
                    continue

                if symmetrical_subs and elasticity2 >= 0:
                    continue

                # WINTRATE_DIFFERENCE_THRESHOLD = 0.05

                chosen_elasticity = elasticity1

                # Ignore the substitutes if the elasticity is too high
                if chosen_elasticity > ELASTICITY_THRESHOLD:
                    continue

                cards_with_subs[card1].append((card2, chosen_elasticity))

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
def regress_inversion_pairs(inversion_pairs,
                            card_data):
    print("Regressing pick rate difference on substitutes seen difference")

    dependent_var = "Pickrate Difference (Card 1 - Card 2)"
    independent_var = "Substitutes Seen Difference (Card 1 - Card 2)"

    regr_data = []

    for pair in inversion_pairs.keys():
        card1, card2 = get_cards_from_pair(pair, card_data)

        if type(card1) is Card:
            card1 = card1.name
        if type(card2) is Card:
            card2 = card2.name

        # Get the coefficient of inversion
        inversion = inversion_pairs[pair]["inversion"]

        # Count the total number of observations of the substitutes
        card1_subs_count = card_data[card1].numberSubstitutes
        card2_subs_count = card_data[card2].numberSubstitutes

        # At this point, card1 is the more picked card
        # because of how find_inversion_pairs works

        # We want the difference between substitutes seen 
        # for the more picked card and the less picked card
        subs_count_diff = card1_subs_count - card2_subs_count

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


def colour_pair_filter(drafts, colours):
    print(f"Filtering drafts by colour pair {colours}")
    print(f"Will parse {len(drafts)} drafts")

    # Eliminate drafs where the drafter didn't end up with at least 6 black cards
    # And at least 6 red cards
    # This is to eliminate drafts where the drafter was not in the right colours
    temp = []

    for draft in drafts:

        num_colour1 = 0
        num_colour2 = 0

        for pick in draft.picks:
            pick_name = cardNamesHash[pick.pick]
            card_obj = name_to_card(pick_name, card_data)
            pick_colour = card_obj.colour

            if pick_colour in colours[0]:
                num_colour1 += 1
            elif pick_colour in colours[1]:
                num_colour2 += 1

        if num_colour1 > COLOURS_THRESHOLD and num_colour2 > COLOURS_THRESHOLD:
            temp.append(draft)

            if debug:
                print(f"Found draft where drafter ended up with \
                    at least {COLOURS_THRESHOLD + 1} {colours[0]} cards \
                    and {COLOURS_THRESHOLD + 1} {colours[1]} cards")

        elif debug:
            print(f"Drater ended up with {num_colour1} {colours[0]} cards \
                    and {num_colour2} {colours[1]} cards, skipping")

    print(f"Number of drafts after eliminating drafts where drafter\n\
            did not end up with at least {COLOURS_THRESHOLD + 1} {colours[0]}\n\
            cards and {COLOURS_THRESHOLD + 1} {colours[1]} cards: {len(temp)}")

    drafts = temp

    print("=====================================")

    if len(drafts) == 0:
        print("No drafts left after filtering by colour pair")
        exit(1)

    # Overwrite the cache to eliminate drafts where the drafter did not end up with at least 6 red and 6 black cards
    with open(os.path.join(os.path.dirname(__file__), "..", "data", "drafts.pickle"), "wb") as f:
        pickle.dump(drafts, f)

    return drafts


def store_availability(card_data,
                       availabilities):

    for availability in availabilities.keys():
        card_name = availability
        card_data[card_name].numberSubstitutes = availabilities[availability]

    return card_data


def log_availabilities(availabilities):
    print("Logging availability of substitutes")

    # Write to a log file
    # Sort all cards by availability score
    # Print card: availability
    # substitute1: elasticity
    # substitute2: elasticity
    # ...
    # ========

    # Sort the cards by availability
    sorted_availabilities = sorted(availabilities.items(), key=lambda x: x[1], reverse=True)

    # Open the log file
    with open(os.path.join(os.path.dirname(__file__), "..", "data", "availabilities.log"), "w") as f:
        # Print each card and its availability
        for card, availability in sorted_availabilities:
            f.write(f"{card}: {sorted_availabilities.index((card, availability))} / {len(sorted_availabilities)} {availability}\n")
            f.write("-----------------------------\n")

            # Sort the substitutes by elasticity ascending
            cards_with_subs[card].sort(key=lambda x: x[1])

            # Print each substitute and its elasticity
            for substitute in cards_with_subs[card]:
                f.write(f"{substitute[0]}: {substitute[1]}\n")

            f.write("=============================\n")


def compute_availability(cards_with_subs,
                         card_data,
                         ):
    print("Computing availability of substitutes")

    # Check if there is a local cache
    # availiabilities.pickle
    # if so return it
    if os.path.exists(os.path.join(os.path.dirname(__file__), "..", "data", "availabilities.pickle")):
        with open(os.path.join(os.path.dirname(__file__), "..", "data", "availabilities.pickle"), "rb") as f:
            availabilities = pickle.load(f)
            card_data = store_availability(card_data, availabilities)
            return availabilities, card_data

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
            sub_availability = abs(card_data[card2].numberSeen * card2_elas / NUM_DRAFTS)
            cardSubstitutesCount += sub_availability

            #cardSubstitutesCount += 1

        availabilities[card] = cardSubstitutesCount

    # Cache the availabilities
    with open(os.path.join(os.path.dirname(__file__), "..", "data", "availabilities.pickle"), "wb") as f:
        pickle.dump(availabilities, f)

    card_data = store_availability(card_data, availabilities)

    return availabilities, card_data


def get_pairs(colours, card_data, rares):
    cards = []

    cards = get_cards_in_colours(card_data, colours, rares)

    pairs = []
    for card1 in cards:
        for card2 in cards:
            if card1 != card2:
                pairs.append((card1, card2))

    # Compute the number of pairs
    num_pairs = len(pairs)
    print(f"Number of pairs: {num_pairs}")

    return cards, pairs


def get_cards_in_colours(card_data, colours, rares):
    cards = []

    all_colours = ["W", "U", "B", "R", "G"]

    other_colours = []
    for colour in all_colours:
        if colour not in colours:
            other_colours.append(colour)

    for card in card_data.values():

        # Cards must contain at least one of the colours
        if not any(colour in card.colour for colour in colours):
            continue

        # Cards must not contain any of the other colours
        if any(colour in card.colour for colour in other_colours):
            continue

        if not rares:
            if card.rarity == "R" or card.rarity == "M":
                continue

        cards.append(card.name)

    print(f"Number of cards in {colours}: {len(cards)}")

    return cards


def print_inversion_pairs(inversion_pairs):
    print("Printing inversion pairs")

    # Sort the inversion pairs by inversion
    sorted_inversion_pairs = sorted(inversion_pairs.items(), key=lambda x: x[1]["inversion"], reverse=True)

    # Print each pair and its inversion
    for pair, pair_info in sorted_inversion_pairs:
        card1, card2 = get_cards_from_pair(pair, card_data)
        print(f"{card1.name} and {card2.name}: {pair_info['inversion']}")

# Print inverted pairs and exit

# Main script


if __name__ != "__main__":
    exit()

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

drafts = parse_drafts(csv_file_path, ltr_cards, NUM_DRAFTS)

print("There are this many drafts in the data set: " + str(len(drafts)))

card_data = getCardData()

colours = ["R", "B"]

cards, pairs = get_pairs(colours, card_data, rares=False)

print("This many pairs of cards: " + str(len(pairs)))

drafts = colour_pair_filter(drafts, colours)

# Go through the drafts, 
# Appending the information we need for the regression to each card
parse_pool_info(drafts)

# Check the pickrate for smite the deathless
seen = 0
picked = 0
for pick in card_data["Smite the Deathless"].picks:
    seen += 1
    if pick["wasPicked"]:
        picked += 1

print(f"Smite the Deathless was seen {seen} times")
print(f"Smite the Deathless was picked {picked} times")
print(f"Smite the Deathless pickrate: {picked / seen}")

# Regress smite the deathless and battle-scarred goblin on Rally at the Hornburg
# With debug on

debug = True

"""
elasticity_substitution("Battle-Scarred Goblin",
                        "Rally at the Hornburg",
                        check_card1_colour=False,
                        check_card2_colour=False,
                        pick_number=False,
                        check_card1_in_pool=True,
                        )

elasticity_substitution("Smite the Deathless",
                        "Battle-Scarred Goblin",
                        check_card1_colour=False,
                        check_card2_colour=False,
                        pick_number=False,
                        check_card1_in_pool=True,
                        )
"""

elasticity_substitution("Smite the Deathless",
                        "Voracious Fell Beast",
                        check_card1_colour=False,
                        check_card2_colour=True,
                        pick_number=False,
                        check_card1_in_pool=False,
                        )

elasticity_substitution("Smite the Deathless",
                        "Éomer of the Riddermark",
                        check_card1_colour=True,
                        check_card2_colour=False,
                        pick_number=False,
                        check_card1_in_pool=False,
                        )

elasticity_substitution("Smite the Deathless",
                        "Éomer of the Riddermark",
                        check_card1_colour=False,
                        check_card2_colour=False,
                        pick_number=False,
                        check_card1_in_pool=False,
                        )


debug = False

# Parameters I've been messing around with:
# 1. check_card1_colour
# 2. weight elasticity by number of observations
# 3. symmetrical_subs
# 4. pick_number
# 5. check_card1_in_pool

cards_with_subs = get_substitutes(cards,
                                  symmetrical_subs=True)

availiabilities, card_data = compute_availability(cards_with_subs, card_data)

log_availabilities(availiabilities)

regress_alsa(cards)

#pairs = compute_pairwise_pickrates(pairs, drafts)


inversion_pairs = find_inversion_pairs(pairs, card_data)

print_inversion_pairs(inversion_pairs)

regress_inversion_pairs(inversion_pairs, card_data)
