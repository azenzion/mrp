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
ELASTICITY_THRESHOLD = 0.0

COLOURS_THRESHOLD = 5
POOL_THRESHOLD = 0.5

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
    pick_for_regr["RinPool"] = 0
    pick_for_regr["BinPool"] = 0
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

    pick_for_regr['colourInPool'] = pick.colourInPool

    pick_for_regr["pick_number"] = pick.pick_number

    pick_for_regr["numCardInPool"] = pick.numCardInPool

    card.picks.append(pick_for_regr)



def dict_increment(dictionary, key):
    if key in dictionary:
        dictionary[key] += 1
    else:
        dictionary[key] = 1


def process_draft_pool(draft):

    # Sort the picks by modified pick number
    # This is the pick number + 15 * pack number + 1
    # This is the order in which the cards were seen

    draft.picks.sort(key=lambda x: x.pick_number + 15 * x.pack_number)




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

        # Ath this point, print the sum of numCardinPool 
        # and the pick number
        if debug:
            print(f"Sum of numCardInPool: {sum(num_card_in_draft_pool.values())}")
            pick_number = pick.pick_number + 1 + 15 * pick.pack_number
            print(f"Pick number: {pick_number} ")

        # Print all the picks 
        # Where there are 4 Easterling Vanguard in pool
        # Relentless Rohirrim is in the pack
        # And Relentless Rohirrim is not picked
        # Print the pick
        # Print the pool
        
        if "Easterling Vanguard" in pick.numCardInPool:
            if pick.numCardInPool["Easterling Vanguard"] == 4:
                # Check if Relentless Rohirrim is in the pack
                rohirrim = False
                for card in pick.pack_cards:
                    if card == cardNumsHash["Relentless Rohirrim"]:
                        rohirrim = True
                        break
                if not rohirrim:
                    #print(f"Rohirrim was picked")
                    # Print openness to red
                    #openness_to_red = calculate_openness_to_colour("R", pick)
                    #print(f"Openness to red: {openness_to_red}")
                    #time.sleep(1)
                    continue

                if pick.pick == cardNumsHash["Relentless Rohirrim"]:
                    continue
                else:
                    print(f"Rohirrim not picked")
                    print(f"Pick was {cardNamesHash[pick.pick]}")
                    #print(f"Pack number was {pick.pack_number}")
                    #print(f"Pick number was {pick.pick_number}")
                    # Print the full pool
                    #print(f"Pool: {pick.numCardInPool}")

                    # Print openness to red
                    openness_to_red = calculate_openness_to_colour("R", pick)
                    print(f"Openness to red: {openness_to_red}")

                    time.sleep(1)


        """
        # Print any pick where the pack number is 2
        # There is a Smite the Deathless in the pool
        # and Voracious Fell Beast is offered
        # Print what the pick was
        if pick.pack_number == 2:
            if "Smite the Deathless" in pick.numCardInPool:
                # Check if Voracious Fell Beast was in the pack
                vor = False
                for card in pick.pack_cards:
                    if card == cardNumsHash["Voracious Fell Beast"]:
                        print(f"Found Voracious Fell Beast in the pack")
                        vor = True
                        break
                if not vor:
                    continue

                if pick.pick == cardNumsHash["Voracious Fell Beast"]:
                    print("Voracious Fell Beast was picked")
                    continue
                else:
                    print("Voracious Fell Beast was not picked")
                    print(f"Found Smite the Deathless in the pool")
                    print(f"Pick was {cardNamesHash[pick.pick]}")
                    print(f"Pack number was {pick.pack_number}")
                    print(f"Pick number was {pick.pick_number}")
                    # Print the full pool
                    print(f"Pool: {pick.numCardInPool}")

                    # Print the openness to black
                    openness_to_black = calculate_openness_to_colour("B", pick)
                    print(f"Openness to black: {openness_to_black}")

                    time.sleep(10)

            """





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
                         num_obs,
                         weight_by_num_obs,
                         eliminate_zero_coef,
                         labels):

    coeff = 0
    for i in range(len(results.params)):
        if debug:
            print(f'coefficient {labels[i]}: {results.params[i]}')

        # Parse the number out of labels[i]
        # If it's not a number, continue
        try:
            num_card2 = int(labels[i].split(" ")[0])
        except ValueError:
            continue

        num_obs_card2 = num_obs[num_card2]

        if debug:
            print(f"Number of observations with {num_card2} substitutes: {num_obs_card2}")

        coeff += results.params[i] * i

    return coeff < 0, coeff


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


def create_exog(picked,
                num_subs,
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
                logify,
                ):

    # Constant term
    exog = np.ones((len(picked), 1))
    labels = ["Constant"]

    if pick_number:
        exog = np.column_stack((exog, pick_number_list))
        labels.append("Pick number")

    if check_card1_colour:
        exog = np.column_stack((exog, card1_colour_count))

        labelname = f"{card1_colour} in pool"

        if logify:
            labelname += " (log)"

        labels.append(labelname)

    if check_card2_colour:
        exog = np.column_stack((exog, card2_colour_count))

        labelname = f"{card2_colour} in pool"

        if logify:
            labelname += " (log)"

        labels.append(labelname)

    if check_card1_in_pool:
        exog = np.column_stack((exog, card1_in_pool))
        labels.append(f"{card1_name} in pool")

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


def calculate_openness_to_colour(colour1,
                                 pick):

    if type(pick) is dict:
        colour_in_pool = pick['colourInPool']
        pick_number = pick['pick_number']
        pack_number = pick['pack_number']
    else:
        try:
            colour_in_pool = pick.colour_in_pool
        except AttributeError:
            colour_in_pool = {}
        pick_number = pick.pick_number
        pack_number = pick.pack_number

    # Calculate openness to colour1
    cards_in_pool = sum(colour_in_pool.values())

    # If no cards_in_pool, fully open to any colour
    if cards_in_pool == 0:
        return 1

    colourshares = {
        'W': 0,
        'U': 0,
        'B': 0,
        'R': 0,
        'G': 0
    }

    for colour in colourshares.keys():
        colour_count = 0
        for key in colour_in_pool.keys():
            if colour in key:
                colour_count += colour_in_pool[key]
        colourshares[colour] = colour_count / cards_in_pool

    share_of_colour1 = colourshares[colour1]

    # eliminate all 0s from colourshares
    colourshares = {k: v for k, v in colourshares.items() if v != 0}

    # get the top 2 colours by share
    top2 = sorted(colourshares.items(), key=lambda x: x[1], reverse=True)[:2]

    # Print the top2 colours
    print(f"Top 2 colours: {top2}")

    # return 1 if colour1 is in the top 2 colours
    for colour in top2:
        if colour1 in colour[0]:
            return 1

    # return 1 if there is only one colour in the pool
    if len(top2) == 1:
        return 1

    # Return 1 minus the percentage of picks that are the second highest colour
    # This is the portion of your pool that you give up
    # to pivot into colour1
    openness = 1 - top2[1][1]

    # Divide by pick_number to reflect the fact that players are less likely to change colours
    # later in the draft
    total_picks = pick_number + 1 + 15 * pack_number

    openness /= total_picks

    return openness

# Count the number of cards in pick.colourInPool
# that are the same colour as card
# excluding cards in excludes
def get_colour_in_pool(pick,
                        card_colour,
                        excludes):
    if card_colour == "R":
        card_colour_in_pool = pick['colourInPool']['R']
    elif card_colour == "B":
        card_colour_in_pool = pick['colourInPool']['B']
    elif card_colour == "BR":
        card_colour_in_pool = pick['colourInPool']['R'] + pick['colourInPool']['B']
    else:
        for exclude in excludes:
            if exclude in pick['numCardInPool']:
                card_colour_in_pool -= pick['numCardInPool'][exclude]

    return card_colour_in_pool


# Filter by number of some key card in the pool
def elasticity_substitution(card1,
                            subs_list,
                            regr_params
                            ):

    debug = False

    if regr_params['debug']:
        debug = True

    logify = regr_params['logify']

    simple = regr_params['simple']

    times_seen = regr_params['times_seen']

    # parse the regression parameters
    check_card1_colour = regr_params['check_card1_colour']
    check_card2_colour = regr_params['check_card2_colour']
    check_card1_in_pool = regr_params['check_card1_in_pool']
    pick_number = regr_params['pick_number']

    if 'pool_threshold' in regr_params:
        pool_threshold = regr_params['pool_threshold']
    else:
        pool_threshold = POOL_THRESHOLD

    # If card1 is passed in the list of substitutes, remove it
    # This is because we *always* use number of card1 already in pool
    # as a control
    if check_card1_in_pool and card1 in subs_list:
        subs_list.remove(card1)

    # Allow function to take a single substitute
    if type(subs_list) is str:
        subs_list = [subs_list]

    # Get colour of cards 1 and 2
    card1_obj = name_to_card(card1, card_data)
    card1_colour = card1_obj.colour

    # Case where there is only one substitute
    card2 = subs_list[0]
    card2_obj = name_to_card(subs_list[0], card_data)
    card2_colour = card2_obj.colour

    # If card1 and card2 are not the same colour
    # Check card2 colour
    if card1_colour == card2_colour:
        check_card2_colour = False

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

    # if card1 and card2 are the same colour, don't check card1 colour or card2 colour
    if card1_colour == card2_colour:
        check_card1_colour = False
        check_card2_colour = False

    for i in range(1, NUM_MULTIPLES + 1):
        num_substitutes[i] = []

    b_in_pool = []
    r_in_pool = []

    if check_card1_colour or check_card2_colour:
        b_in_pool = [x['colourInPool']['B'] if 'B' in x['colourInPool'] else 0 for x in card1_obj.picks]
        r_in_pool = [x['colourInPool']['R'] if 'R' in x['colourInPool'] else 0 for x in card1_obj.picks]

    picked = [x['wasPicked'] for x in card1_obj.picks]

    if check_card1_in_pool:
        card1_in_pool = [x['sameCardInPool'] for x in card1_obj.picks]

    if not simple:
        for pick in card1_obj.picks:
            num_substitutes = count_substitutes(pick,
                                                subs_list,
                                                num_substitutes,
                                                card1)

        # Eliminate values that didn't have enough observations
        num_substitutes, num_observations = eliminate_low_observations(num_substitutes)

    simple_num_substitutes = [x['numCardInPool'][card2] if card2 in x['numCardInPool'] else 0 for x in card1_obj.picks]

    if logify:
        simple_num_substitutes = [np.log(x + 1) for x in simple_num_substitutes]

    if check_card1_colour:
        if card1_colour == "R":
            card1_colour_count = r_in_pool
        elif card1_colour == "B":
            card1_colour_count = b_in_pool
        elif card1_colour == "BR":
            card1_colour_count = [x + y for x, y in zip(r_in_pool, b_in_pool)]

    # replace card1_colour_count with openness to colour1
    # a player is 1 if they are not committed to 2 colours
    # and the portion of the pool that is colour1 if they are
    card1_colour_count = [calculate_openness_to_colour(card1_colour, pick) for pick in card1_obj.picks]

    # Print the first 50 values of card1_colour_count
    if debug:
        print(f"First 50 values of {card1} colour count:")
        print(card1_colour_count[:50])

    # Do the same for card2
    if check_card2_colour:
        if card2_colour == "R":
            card2_colour_count = r_in_pool
        elif card2_colour == "B":
            card2_colour_count = b_in_pool
        elif card2_colour == "BR":
            card2_colour_count = [x + y for x, y in zip(r_in_pool, b_in_pool)]

    # Create the pick_number array
    pick_number_list = []

    pick_number_list = [15 * x['pack_number'] + x['pick_number'] + 1 for x in card1_obj.picks]

    # Create a variable that represents the number of times you are seeing the card this draft
    times_seen = [0 if x['pick_number'] < 8 else 1 for x in card1_obj.picks]

    # Create the matrices for the regression
    if not simple:

        if debug:
            for i in range(1, len(num_substitutes.keys())):
                print(f"Number of times {card1} was on offer with {i} {card2} in pool: {sum(num_substitutes[i])}")

        endog = np.array(picked)

        exog, labels = create_exog(picked,
                                num_substitutes,
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
                                card1,
                                logify)

        if times_seen:
            exog = np.column_stack((exog, times_seen))
            labels.append("Times seen")

        # Create the model
        model = sm.OLS(endog, exog)

        model.exog_names[:] = labels

        results = model.fit()

        substitutes, elasticity = check_if_substitutes(results,
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

    # Simple model
    # Continuous on number of cards in pool
    if simple:

        endog = picked

        # Initialize exog to a column of ones
        # this is the constant term
        exog = np.ones((len(simple_num_substitutes), 1))
        labels = [f"{card1} baseline"]

        # Add the number of substitutes in the pool
        #exog = np.column_stack((exog, simple_num_substitutes))
        #labels.append(f"{card2} in pool")
        #if logify:
        #    labels[-1] += " (log)"

        # Create the model
        model = sm.OLS(endog, exog)

        if check_card1_in_pool:
            exog = np.column_stack((exog, card1_in_pool))
            labels.append(f"{card1} in pool")

        if check_card1_colour:
            exog = np.column_stack((exog, card1_colour_count))
            labels.append(f"Opennness to {card1_colour}")

        if check_card2_colour:
            exog = np.column_stack((exog, card2_colour_count))
            labels.append(f"{card2_colour} in pool")

        if pick_number:
            exog = np.column_stack((exog, pick_number_list))
            labels.append("Pick number")

        if times_seen:
            exog = np.column_stack((exog, times_seen))
            labels.append("Times seen")

        # print number of parameters
        if debug:
            print(labels)

        # Print width of exog
        if debug:
            print(f"Width of exog: {len(exog[0])}")

        # Label and run the model
        model = sm.OLS(endog, exog)

        model.exog_names[:] = labels

        results = model.fit()

        elasticity = results.params[1]

        # get the parameter on simple_num_substitutes
        # if it's negative, they are substitutes
        # if it's positive, they are complements
        # if it's 0, they are independent
        if debug:
            print(results.summary())

        if elasticity < 0:
            substitutes = True
        else:
            substitutes = False

        if debug:
            print(f"Elasticity of substitution for {card1} and {subs_list}: {elasticity}")

            if substitutes:
                print(f"{card1} and {subs_list} are substitutes")
            else:
                print(f"{card1} and {subs_list} are not substitutes")


    return elasticity


# Regress all cards ALSA on the number of subsitutes seen in the sample
# and on their GIH winrate
# Takes a list of cards
def regress_alsa(cards, availabilities):
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
            availability_score = availabilities[card]
            #number_substitutes = np.log(availability_score + 1)
            number_substitutes = availability_score
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

    # Create the simple regression frame
    # no number of substitutes
    simple_regression_data = []
    for card in cards:
        card_obj = name_to_card(card, card_data)
        gih_winrate = card_obj.gameInHandWinrate
        alsa = card_obj.alsa

        simple_regression_data.append([card,
                                gih_winrate,
                                alsa])
        
    # Create the simple data frame
    simple_df = pd.DataFrame(simple_regression_data, columns=["card_name",
                                                "gih_winrate",
                                                "alsa"])
    
    # Run the simple regression
    # The model is:
    # alsa = b0 + b1 * gih_winrate

    # Create the model
    simple_model = sm.OLS.from_formula("alsa ~ gih_winrate",
                                data=simple_df)
    
    # Fit the model
    simple_results = simple_model.fit()

    # Print the results
    print(simple_results.summary())


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


def is_substitute_for(card,
                      sub,
                        cards_with_subs):

    for cardname, elasticity in cards_with_subs[card]:
        if cardname == sub:
            return True

    return False


# Apply a series of tests to this substitute set 
# To see how well it matches intuition
def test_substitutes_set(cards_with_subs,
                         availabilities,
                         regr_params):

    success_count = 0
    fail_count = 0

    filename = "substitutes_test_results"
    filename += suffix_params(regr_params)
    filename += ".txt"

    print(f"Writing results to {filename}")

    # As we go, print and write to a file
    with open(os.path.join(os.path.dirname(__file__), "..", "data", filename), "w") as f:

        print(type(availabilities))

        if type(availabilities) is not dict:
            availabilities = availabilities[0]
        # Rally at the Hornburg has more substitutes than Smite the Deathless
        # Is the hypothesis true
        # Check if Rally at the Hornburg has more substitutes than Smite the Deathless
        if availabilities["Rally at the Hornburg"] > availabilities["Smite the Deathless"]:
            f.write("success: Rally at the Hornburg has more substitutes than Smite the Deathless\n")
            success_count += 1
        else:
            f.write("fail: Rally at the Hornburg has fewer substitutes than Smite the Deathless\n")
            fail_count += 1

        # Same colour similar
        # Battle-Scarred Goblin should be a substitute for Rally at the Hornburg
        f.write("Same-Colour Similar")
        f.write("=====================================")
        if is_substitute_for("Rally at the Hornburg", "Battle-Scarred Goblin", cards_with_subs):
            f.write("success: Battle-Scarred Goblin is a substitute for Rally at the Hornburg\n")
            success_count += 1
        else:
            f.write("fail: Battle-Scarred Goblin is not a substitute for Rally at the Hornburg\n")
            fail_count += 1

        # Rohirrim Lancer and Rally at the Hornburg should be substitutes
        if is_substitute_for("Rally at the Hornburg", "Rohirrim Lancer", cards_with_subs):
            f.write("success: Rohirrim Lancer is a substitute for Rally at the Hornburg\n")
            success_count += 1
        else:
            f.write("fail: Rohirrim Lancer is not a substitute for Rally at the Hornburg\n")
            fail_count += 1

        if is_substitute_for("Relentless Rohirrim", "Olog-hai Crusher", cards_with_subs):
            print("Olog-hai Crusher is a substitute for Relentless Rohirrim")
            f.write("success: Olog-hai Crusher is a substitute for Relentless Rohirrim\n")
            success_count += 1
        else:
            print("Olog-hai Crusher is not a substitute for Relentless Rohirrim")
            f.write("fail: Olog-hai Crusher is not a substitute for Relentless Rohirrim\n")
            fail_count += 1

        if is_substitute_for("Relentless Rohirrim", "Warbeast of Gorgoroth", cards_with_subs):
            print("Warbeast of Gorgoroth is a substitute for Relentless Rohirrim")
            f.write("success: Warbeast of Gorgoroth is a substitute for Relentless Rohirrim\n")
            success_count += 1
        else:
            print("Warbeast of Gorgoroth is not a substitute for Relentless Rohirrim")
            f.write("fail: Warbeast of Gorgoroth is not a substitute for Relentless Rohirrim\n")
            fail_count += 1

        f.write("=====================================")
        f.write("Same Colour Different")

        # Same Colour different cards should not be substitutes
        # Smite the Deathless and Warbeast of Gargaroth should not be substitutes
        if is_substitute_for("Smite the Deathless", "Warbeast of Gorgoroth", cards_with_subs):
            f.write("fail: Warbeast of Gorgoroth is a substitute for Smite the Deathless\n")
            fail_count += 1
        else:
            f.write("success: Warbeast of Gorgoroth is not a substitute for Smite the Deathless\n")
            success_count += 1

        # cross-colour different cards should not be substitutes
        f.write("=====================================")
        f.write("Cross-Colour Different")

        # Cross-Colour different cards should not be substitutes
        # Voracious Fell Beast should not be a substitute for Smite the Deathless
        if is_substitute_for("Smite the Deathless", "Voracious Fell Beast", cards_with_subs):
            f.write("fail: Voracious Fell Beast is a substitute for Smite the Deathless\n")
            fail_count += 1
        else:
            f.write("success: Voracious Fell Beast is not a substitute for Smite the Deathless\n")
            success_count += 1

        if is_substitute_for("Relentless Rohirrim", "Easterling Vanguard", cards_with_subs):
            print("Easterling Vanguard is a substitute for Relentless Rohirrim")
            f.write("fail: Easterling Vanguard is a substitute for Relentless Rohirrim\n")
            fail_count += 1
        else:
            print("Easterling Vanguard is not a substitute for Relentless Rohirrim")
            f.write("success: Easterling Vanguard is not a substitute for Relentless Rohirrim\n")
            success_count += 1

        if is_substitute_for("Relentless Rohirrim", "Mordor Muster", cards_with_subs):
            print("Mordor Muster is a substitute for Relentless Rohirrim")
            f.write("fail: Mordor Muster is a substitute for Relentless Rohirrim\n")
            fail_count += 1
        else:
            print("Mordor Muster is not a substitute for Relentless Rohirrim")
            f.write("success: Mordor Muster is not a substitute for Relentless Rohirrim\n")
            success_count += 1

        # Cross-colour similar cards should be substitutes
        f.write("Cross-Colour Similar")
        f.write("=====================================")
        if is_substitute_for("Battle-Scarred Goblin", "Easterling Vanguard", cards_with_subs):
            f.write("success: Easterling Vanguard is a substitute for Battle-Scarred Goblin\n")
            success_count += 1
        else:
            f.write("fail: Easterling Vanguard is not a substitute for Battle-Scarred Goblin\n")
            fail_count += 1

        # Book of Mazarbul and Dunland Crebain should be substitutes
        if is_substitute_for("Book of Mazarbul", "Dunland Crebain", cards_with_subs):
            f.write("success: Dunland Crebain is a substitute for Book of Mazarbul\n")
            success_count += 1
        else:
            f.write("fail: Dunland Crebain is not a substitute for Book of Mazarbul\n")
            fail_count += 1

        if is_substitute_for("Relentless Rohirrim", "Grond, the Gatebreaker", cards_with_subs):
            print("Grond, the Gatebreaker is a substitute for Relentless Rohirrim")
            f.write("success: Grond, the Gatebreaker is a substitute for Relentless Rohirrim\n")
            success_count += 1
        else:
            print("Grond, the Gatebreaker is not a substitute for Relentless Rohirrim")
            f.write("fail: Grond, the Gatebreaker is not a substitute for Relentless Rohirrim\n")
            fail_count += 1

        if is_substitute_for("Relentless Rohirrim", "Snarling Warg", cards_with_subs):
            print("Snarling Warg is a substitute for Relentless Rohirrim")
            f.write("success: Snarling Warg is a substitute for Relentless Rohirrim\n")
            success_count += 1
        else:
            print("Snarling Warg is not a substitute for Relentless Rohirrim")
            f.write("fail: Snarling Warg is not a substitute for Relentless Rohirrim\n")
            fail_count += 1

        f.write("=====================================")
        f.write("Same-Colour Removal Spells")
        # Same-colour removal spells should be substitutes
        count = 0
        for card in redRemoval:
            if is_substitute_for("Smite the Deathless", card, cards_with_subs):
                count += 1

        f.write(f"Found {count} out of {len(redRemoval)} red removal spells as substitutes for Smite the Deathless\n")

        # Different-colour removal spells should be substitutes
        count = 0
        blackRemoval = ["Claim the Precious",
                        "Bitter Downfall",
                        "Gollum's Bite",
                        "Lash of the Balrog",]
        for card in blackRemoval:
            if is_substitute_for("Smite the Deathless", card, cards_with_subs):
                count += 1
        f.write(f"Found {count} out of {len(blackRemoval)} black removal spells as substitutes for Smite the Deathless\n")

        f.write("=====================================")
        f.write("Cross-Colour Removal Spells")

        # The black removal spells should be substitutes for Smite the Deathless
        count = 0
        for card in blackRemoval:
            if is_substitute_for("Smite the Deathless", card, cards_with_subs):
                count += 1
        f.write(f"Found {count} out of {len(blackRemoval)} black removal spells as substitutes for Smite the Deathless\n")

        if is_substitute_for("Relentless Rohirrim", "Smite the Deathless", cards_with_subs):
            print("Smite the Deathless is a substitute for Relentless Rohirrim")
            f.write("success: Smite the Deathless is a substitute for Relentless Rohirrim\n")
            success_count += 1
        else:
            print("Smite the Deathless is not a substitute for Relentless Rohirrim")
            f.write("fail: Smite the Deathless is not a substitute for Relentless Rohirrim\n")
            fail_count += 1

        if is_substitute_for("Relentless Rohirrim", "Battle-Scarred Goblin", cards_with_subs):
            print("Battle-Scarred Goblin is a substitute for Relentless Rohirrim")
            f.write("fail: Battle-Scarred Goblin is a substitute for Relentless Rohirrim\n")
            fail_count += 1
        else:
            print("Battle-Scarred Goblin is not a substitute for Relentless Rohirrim")
            f.write("success: Battle-Scarred Goblin is not a substitute for Relentless Rohirrim\n")
            success_count += 1

        # Print summary
        print(f"Number of successes: {success_count} / {success_count + fail_count}")
        print(f"Number of failures: {fail_count} / {success_count + fail_count}")
        f.write(f"Number of successes: {success_count} / {success_count + fail_count}\n")
        f.write(f"Number of failures: {fail_count} / {success_count + fail_count}\n")

        # return the number of successes
        return success_count


# Get all the substitutes for a list of cards
# Takes a list of cards
# Returns {
#  card1: [(sub1, elasticity1), (sub2, elasticity2), ...],
#  card2: ...
# }
def get_substitutes(cards,
                    regr_params,
                    refresh=False) -> dict:
    print("Getting substitutes for a list of cards")

    symmetrical_subs = regr_params['symmetrical_subs']

    filename = "substitutes"
    filename += suffix_params(regr_params)
    filename += ".pickle"

    # If refresh, delete any local cache
    if refresh:
        if os.path.exists(os.path.join(os.path.dirname(__file__), "..", "data", filename)):
            os.remove(os.path.join(os.path.dirname(__file__), "..", "data", filename))

    # If there is a local cache, use it
    if os.path.exists(os.path.join(os.path.dirname(__file__), "..", "data", filename)):
        print("Found local cache of substitutes")
        with open(os.path.join(os.path.dirname(__file__), "..", "data", filename), "rb") as f:
            return pickle.load(f)

    # If there is no local cache, compute the substitutes
    cards_with_subs = {}

    cards_with_elasticities = {}

    completion_times = []

    print(f"Computing substitutes for regression parameters: {regr_params}")
    # Compute the elasticity of substitution for each pair of cards
    for card1 in cards:
        print("Getting substututes for " + card1)
        print(f"Card is number {cards.index(card1) + 1} out of {len(cards)}")
        print(f"Estimated time remaining: {np.mean(completion_times) * (len(cards) - cards.index(card1))} seconds")

        start_time = time.time()
        for card2 in cards:
            if card1 != card2:

                # pass in the regr_params as boolean vars
                elasticity = elasticity_substitution(card1,
                                                     card2,
                                                     regr_params)
                cards_with_elasticities[card1, card2] = elasticity

        completion_times.append(time.time() - start_time)

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
    with open(os.path.join(os.path.dirname(__file__), "..", "data", filename), "wb") as f:
        pickle.dump(cards_with_subs, f)

    # If we recalculated substitutes, need to delete availabilities
    # as these are dependedent
    if refresh:
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



def log_availabilities(availabilities, cards_with_subs, regr_params):
    print("Logging availability of substitutes")

    # Write to a log file
    # Sort all cards by availability score
    # Print card: availability
    # substitute1: elasticity
    # substitute2: elasticity
    # ...
    # ========

    # Sort the cards by availability
    sorted_availabilities = sorted(availabilities.items(),
                                   key=lambda x: x[1],
                                   reverse=True)

    # Create the filename, based on the regression parameters
    filename = "availabilities"

    filename += suffix_params(regr_params)

    # Open the log file
    with open(os.path.join(os.path.dirname(__file__), "..", "data", filename + ".log"), "w") as f:
        # Log the regression parameters
        f.write("Regression parameters\n")
        f.write("======================\n")
        f.write(f"check_card1_colour: {regr_params['check_card1_colour']}\n")
        f.write(f"check_card2_colour: {regr_params['check_card2_colour']}\n")
        f.write(f"check_card1_in_pool: {regr_params['check_card1_in_pool']}\n")
        f.write(f"pick_number: {regr_params['pick_number']}\n")
        f.write("======================\n")


        # Print each card and its availability
        for card, availability in sorted_availabilities:
            f.write(f"Card {card}\n")
            f.write(f"Rank {sorted_availabilities.index((card, availability))} out of {len(sorted_availabilities)}\n")
            f.write(f"availability score: {availability}\n")
            f.write("-----------------------------\n")

            # Sort the substitutes by elasticity ascending
            cards_with_subs[card].sort(key=lambda x: x[1])

            # Print each substitute and its elasticity
            for substitute in cards_with_subs[card]:
                f.write(f"{substitute[0]}: {substitute[1]}\n")

            f.write("=============================\n")


def compute_availability(cards_with_subs,
                         card_data,
                         regr_params,
                         ):
    print("Computing availability of substitutes")

    filename = "availabilities"
    filename += suffix_params(regr_params)
    filename += ".pickle"

    # Check if there is a local cache
    # availiabilities.pickle
    # if so return it
    if os.path.exists(os.path.join(os.path.dirname(__file__), "..", "data", filename)):
        print("Found local cache of availabilities")
        with open(os.path.join(os.path.dirname(__file__), "..", "data", filename), "rb") as f:
            return pickle.load(f), card_data
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
    with open(os.path.join(os.path.dirname(__file__), "..", "data", filename), "wb") as f:
        pickle.dump(availabilities, f)

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


def suffix_to_regr_params(suffix):
    regr_params = {
        "check_card1_colour": False,
        "check_card2_colour": False,
        "pick_number": False,
        "check_card1_in_pool": False,
        "symmetrical_subs": False,
        "logify": False,
        "debug": False,
        "simple": False,
    }

    if "colour1" in suffix:
        regr_params['check_card1_colour'] = True
    if "colour2" in suffix:
        regr_params['check_card2_colour'] = True
    if "same_in_pool" in suffix:
        regr_params['check_card1_in_pool'] = True
    if "pick_number" in suffix:
        regr_params['pick_number'] = True
    if "log" in suffix:
        regr_params['logify'] = True
    if "sym" in suffix:
        regr_params['symmetrical_subs'] = True
    if "simple" in suffix:
        regr_params['simple'] = True

    return regr_params





def suffix_params(regr_params):
    filename = ""
    if regr_params['check_card1_colour']:
        filename += "_colour1"
    if regr_params['check_card2_colour']:
        filename += "_colour2"
    if regr_params['check_card1_in_pool']:
        filename += "_same_in_pool"
    if regr_params['pick_number']:
        filename += "_pick_number"

    # Logify
    if regr_params['logify']:
        filename += "_log"

    # Symmetrical or asymmetrical substitutes
    if regr_params['symmetrical_subs']:
        filename += "_sym"

    # Simple model
    if regr_params['simple']:
        filename += "_simple"

    if regr_params['times_seen']:
        filename += "_times_seen"

    return filename



def generate_subs_groupings(cards,
                            card_data,
                            regr_params,
                            refresh):

    # Print the suffix
    print(suffix_params(regr_params))

    # Get the substitutes for each card
    cards_with_subs = get_substitutes(cards, regr_params, refresh)

    # Compute the availability of each card
    availabilities, card_data = compute_availability(cards_with_subs,
                                                     card_data,
                                                     regr_params)

    # Log the availabilities
    log_availabilities(availabilities, cards_with_subs, regr_params)

    return cards_with_subs, availabilities, card_data


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

# Set the actual number of drafts
NUM_DRAFTS = len(drafts)

# Go through the drafts,
# Appending the information we need for the regression to each card
parse_pool_info(drafts)

# Define the parameters for the elasticity regression
regr_params = {
    "check_card1_colour": False,
    "check_card2_colour": False,
    "pick_number": False,
    "check_card1_in_pool": False,
    "symmetrical_subs": True,
    "logify": True,
    "debug": False,
    "simple": True,
}


bools = [False, True]
# Run on Easterling Vanguard and Battle-Scarred Goblin
# Simple and not Simple

# Baseline Model
regr_params['simple'] = True
regr_params['logify'] = False
regr_params['times_seen'] = True
regr_params['check_card1_colour'] = True
regr_params['pick_number'] = False
regr_params['debug'] = True

#for i in range(20):
#    POOL_THRESHOLD = 0.05 * i

# Do the relentless Rohirrim Tests
elasticity_substitution("Relentless Rohirrim", "Easterling Vanguard", regr_params)


exit()

# Run the model on combinations of parameters
good_runs = []

for check_card1_colour in bools:
    for logify in bools:
        for pick_number in bools:
            regr_params['logify'] = logify
            regr_params['check_card1_colour'] = check_card1_colour

            cards_with_subs, availabilities, card_data = generate_subs_groupings(cards, card_data, regr_params, refresh=False)

            # Test the substitutes set  
            # If it passes all the tests, add it to the list of good runs
            success_count = test_substitutes_set(cards_with_subs, availabilities, regr_params)
            if success_count >= 10:
                good_runs.append((suffix_params(regr_params), success_count))


# Sort the good runs by number of successes
good_runs.sort(key=lambda x: x[1], reverse=True)
# print each, with the number of successes
for run in good_runs:
    print(f"{run[0]}: {run[1]}")

exit()
# Print the largest substution effects
# And the larges complement effects

substitutions = []
for card in cards_with_subs.keys():
    for sub, elasticity in cards_with_subs[card]:
        substitutions.append((card, sub, elasticity))

substitutions.sort(key=lambda x: x[2])

print("Largest substitution effects")
for i in range(1, 11):
    substitution = substitutions[i - 1]
    print(f"{substitution[0]} and {substitution[1]}: {substitution[2]}")

regress_alsa(cards, availabilities)

pairs = compute_pairwise_pickrates(pairs, drafts)


inversion_pairs = find_inversion_pairs(pairs, card_data)

print_inversion_pairs(inversion_pairs)

regress_inversion_pairs(inversion_pairs, card_data)
