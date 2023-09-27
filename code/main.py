# Import the draft data
import os
import csv
import time
from filecache import filecache
from statistics import mean
import numpy as np
import statsmodels.api as sm


csv_file_path = os.path.join(os.path.dirname(__file__), "..", "data", "draft_data_public.LTR.PremierDraft.csv")
cardlist_file_path = os.path.join(os.path.dirname(__file__), "..", "data", "ltr_cards.txt")

# TODO: Better naming conventions for cache files
debug = False

num_drafts = 10000

def getColours():
    COLOURS = ['W', 'U', 'B', 'R', 'G', '']

    tempColours = []

    for COLOUR in COLOURS:
        if COLOUR not in tempColours:
            tempColours.append(COLOUR)
        for secondColour in COLOURS:
            if COLOUR != secondColour:
                tempColours.append(COLOUR + secondColour)
            for thirdColour in COLOURS:
                if COLOUR != secondColour and COLOUR != thirdColour and secondColour != thirdColour:
                    tempColours.append(COLOUR + secondColour + thirdColour)
                for fourthColour in COLOURS:
                    if COLOUR != secondColour and COLOUR != thirdColour and COLOUR != fourthColour and secondColour != thirdColour and secondColour != fourthColour and thirdColour != fourthColour:
                        tempColours.append(COLOUR + secondColour + thirdColour + fourthColour)
                    for fifthColour in COLOURS:
                        if COLOUR != secondColour and COLOUR != thirdColour and COLOUR != fourthColour and COLOUR != fifthColour and secondColour != thirdColour and secondColour != fourthColour and secondColour != fifthColour and thirdColour != fourthColour and thirdColour != fifthColour and fourthColour != fifthColour:
                            tempColours.append(COLOUR + secondColour + thirdColour + fourthColour + fifthColour)
    COLOURS = tempColours
    return COLOURS

COLOURS = getColours()

# Magic numbers
NUM_HEADERS = 12
END_OF_PACK = 277
END_OF_POOL = 2 * END_OF_PACK - NUM_HEADERS - 1

# Dawn of a new age is at the end for some reason
DAWN_OF_A_NEW_AGE_PACK = END_OF_POOL + 1
DAWN_OF_A_NEW_AGE_POOL = END_OF_POOL + 2

class Draft:

    def __init__(self):
        self.draft_id = ""
        self.picks = []

class Pick:

    def __init__(self):
        self.expansion = ""
        self.event_type = ""
        self.draft_id = ""
        self.draft_time = ""
        self.rank = ""
        self.event_match_wins = ""
        self.event_match_losses = ""
        self.pack_number = ""
        self.pick_number = ""
        self.pick = ""
        self.pick_maindeck_rate = ""
        self.pick_sideboard_in_rate = ""

        self.pack_cards = []
        self.pool_cards = []

        self.numCardInPool = {}

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


def getPairPickRatesFromCache():

    pairs = {}

    # Read in the pair pick rates from the cache
    with open(os.path.join(os.path.dirname(__file__), "..", "data", "pick_rates.csv"), "r") as f:
        csv_reader = csv.reader(f, delimiter=",")
        next(csv_reader)
        for row in csv_reader:
            pairName = row[0] + " & " + row[1]
            pairs[pairName] = [float(row[2]), float(row[3])]

    return pairs

def computePairwisePickSubstitutes(pairs, card_data):
    # Eliminate every pair where there's a GIH winrate of 0
    validPairs = {}
    for pair in pairs.keys():
        card1, card2 = getCardsFromPair(pair, card_data)
        if pairs[pair][0] == 0.0 or pairs[pair][1] == 0.0:
            print (f"Eliminating {pair} because one of them had a 0 pick rate in the data")
            continue
        elif card1.gameInHandWinrate == 0.0 or card2.gameInHandWinrate == 0.0:
            print (f"Eliminating {pair} because one of them had a 0 GIH winrate in the data")
            continue
        else:
            validPairs[pair] = pairs[pair]

    # For each pair, caclulate the absolute difference in pick rate
    for pair in validPairs.keys():
        card1, card2 = getCardsFromPair(pair, card_data)

        card1PickRate = pairs[pair][0]
        card2PickRate = pairs[pair][1]
    
        # Calculate the absolute difference in pick rate
        absDiff = abs(card1PickRate - card2PickRate)

        # Store the absolute difference in the pair
        validPairs[pair] = [card1PickRate, card2PickRate, absDiff]

    # Sort the pairs by absolute difference in pick rate
    sortedPairs = sorted(validPairs.items(), key=lambda x: x[1][2], reverse=True)


    # For each card, compute the substitutes
    # We define these as cards where, when players have to choose between the two cards,
    # The absolute difference in pick rate is less than 5%
    for pair in sortedPairs:
        card1, card2 = getCardsFromPair(pair[0], card_data)
        if pair[1][2] < 0.05:
            print(f"Adding {card1.name} and {card2.name} as substitutes. pickrates: {pair[1][0]} {pair[1][1]}")
            card1.substitutes.append(card2)
            card2.substitutes.append(card1)


    # Order cards by number of substitutes
    cardSubstitutes = {}
    for card in card_data.values():
        cardSubstitutes[card.name] = len(card.substitutes)

    # Sort the cards by number of substitutes
    sortedSubstitutes = sorted(cardSubstitutes.items(), key=lambda x: x[1], reverse=True)

    # Eliminate cards with no substitutes
    cardsWithSubstitutes = {}
    for cardName in sortedSubstitutes:
        card = getCardFromCardName(cardName[0], card_data)
        if len(card.substitutes) > 0:
            cardsWithSubstitutes[cardName[0]] = cardName[1]


    # Print the sorted cards
    for cardName in sortedSubstitutes:

        if cardName[1] == 0:
            continue

        card = getCardFromCardName(cardName[0], card_data)
        print(f"{cardName[0]}: {cardName[1]}")
        for substitute in card.substitutes:
            print(f"    {substitute.name}")

    return cardsWithSubstitutes

def getCardFromCardName(cardName, card_data):
    if cardName in card_data:
        return card_data[cardName]

    print("ERROR: Card not found in card_data")
    print(f"Card: {cardName}")
    exit(1)

def getCardsFromPair(pair, card_data):
    card1Name = pair.split(" & ")[0]
    card2Name = pair.split(" & ")[1]

    card1 = getCardFromCardName(card1Name, card_data)
    card2 = getCardFromCardName(card2Name, card_data)

    if not card1 or not card2:
        print("ERROR: Card not found in card_data")
        print(f"Card 1: {card1Name}")
        print(f"Card 2: {card2Name}")
        exit(1)

    return card1, card2

def readInversionPairsFromCache():
    inversionPairs = {}
    with open(os.path.join(os.path.dirname(__file__), "..", "data", "inversion_pairs.csv"), "r") as f:
        csv_reader = csv.reader(f, delimiter=",")
        next(csv_reader)
        for row in csv_reader:
            pairName = row[0] + " & " + row[1]
            inversionPairs[pairName] = [float(row[2]), float(row[3]), float(row[4]), float(row[5])]
    return inversionPairs

def parsePercentage(percentage):
    if percentage == "":
        return 0.0
    else:
        if 'pp' in percentage:
            return float(percentage[:-2])
        else:
            return float(percentage[:-1])

def countSubstitutes(drafts, card1, card2):

    # Look for drafts where a player chose between card1 and card2
    choiceCount = 0
    card1Count = 0
    card2Count = 0
    substituteDrafts = []
    for draft in drafts:
        for pick in draft.picks:
            if card1 in pick.pack_cards and card2 in pick.pack_cards:
                if debug:
                    print(f"Pick has {card1} and {card2}")

                # Check if the player picked one of the two cards
                if pick.pick == card1 or pick.pick == card2:
                    if debug:
                        print(f"Player picked {pick.pick}")
                    choiceCount += 1
                
                # Check whether they picked card1 or card2
                if pick.pick == card1:
                    card1Count += 1
                if pick.pick == card2:
                    card2Count += 1

                substituteDrafts.append(draft)

    if choiceCount == 0:
        return 0.0, 0.0

    if debug:
        # Print the number of drafts with the choice
        print(f"Number of picks with the choice: {choiceCount}")

        print(f"Number of times they picked {card1}: {card1Count}")
        print(f"Number of times they picked {card2}: {card2Count}")
        
        # Print percentage players chose each card
        print(f"Percentage of times they picked {card1}: {card1Count / choiceCount}")
        print(f"Percentage of times they picked {card2}: {card2Count / choiceCount}")

    return card1Count / choiceCount, card2Count / choiceCount

def parseDrafts(csv_file_path, ltr_cards, numDrafts=1000):
    csv_reader = None
    with open(csv_file_path, "r") as f:
        # Create a CSV reader
        csv_reader = csv.reader(f, delimiter=",")
    
        header_row = next(csv_reader)
        
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
            pick.expansion = row[0]
            pick.event_type = row[1]
            pick.draft_id = row[2]
            pick.draft_time = row[3]
            pick.rank = row[4]
            pick.event_match_wins = row[5]
            pick.event_match_losses = row[6]
            pick.pack_number = row[7]
            pick.pick_number = row[8]
            pick.pick = row[9]
            pick.pick_maindeck_rate = row[10]
            pick.pick_sideboard_in_rate = row[11]

            # Parse the cards
            for i in range(NUM_HEADERS, END_OF_PACK):
                if row[i] == "1":
                    pick.pack_cards.append(ltr_cards[i - NUM_HEADERS])
        
            # Parse the pool
            for i in range(END_OF_PACK, END_OF_POOL):
                num_card_in_pool = row[i]
                for k in range(int(num_card_in_pool)):
                    pick.pool_cards.append(ltr_cards[i - END_OF_PACK])

            # Dawn of a new hope
            if row[DAWN_OF_A_NEW_AGE_PACK] == "1":
                pick.pack_cards.append("Dawn of a New Age")
            if row[DAWN_OF_A_NEW_AGE_POOL] == "1":
                pick.pool_cards.append("Dawn of a New Age")

            if debug:
                print("PACK")
                for x, card in enumerate(pick.pack_cards):
                    print(f"{x} card: {card}")

                print("POOL")
                for x, card in enumerate(pick.pool_cards):
                    print(f"{x} card: {card}")


            if pick.draft_id == current_draft.draft_id:
                if debug:
                    print(f"Appending pick {pick.pick_number} to current draft")

                current_draft.picks.append(pick)
                
            else:
                if debug:
                    print(f"Appending pick {pick.pick_number} to new draft {draft.draft_id}")

                draft = Draft()
                draft.draft_id = pick.draft_id
                drafts.append(draft)
                current_draft = draft
                current_draft.picks.append(pick)

            # Stop after num_drafts drafts
            if len(drafts) > numDrafts:
                break
    
    # Cache to disk
    with open(os.path.join(os.path.dirname(__file__), "..", "data", f"drafts.csv"), "w") as f:
        csv_writer = csv.writer(f, delimiter=",")
        csv_writer.writerow(["Expansion", "Event Type", "Draft ID", "Draft Time", "Rank", "Event Match Wins", "Event Match Losses", "Pack Number", "Pick Number", "Pick", "Pick Maindeck Rate", "Pick Sideboard In Rate", "Pack Cards", "Pool Cards"])
        for draft in drafts:
            for pick in draft.picks:
                csv_writer.writerow([pick.expansion, pick.event_type, pick.draft_id, pick.draft_time, pick.rank, pick.event_match_wins, pick.event_match_losses, pick.pack_number, pick.pick_number, pick.pick, pick.pick_maindeck_rate, pick.pick_sideboard_in_rate, pick.pack_cards, pick.pool_cards])
    
    return drafts

def findInversionPairs(pairs, card_data):
    inversionPairs = {}
    # Narrow down to pairs where the card with the lower GIH winrate is picked more often
    for pair in pairs.keys():
        card1, card2 = getCardsFromPair(pair, card_data)

        card1PickRate = pairs[pair][0]
        card2PickRate = pairs[pair][1]

        if card1.gameInHandWinrate < card2.gameInHandWinrate:
            if card1PickRate > card2PickRate:
                inversionPairs[pair] = [card1PickRate, card2PickRate, card1.gameInHandWinrate, card2.gameInHandWinrate]
        elif card2PickRate > card1PickRate:
                inversionPairs[pair] = [card1PickRate, card2PickRate, card1.gameInHandWinrate, card2.gameInHandWinrate]

    # Print the number of inversion pairs
    print(f"Number of inversion pairs: {len(inversionPairs.keys())}")
    # Print the inversion pairs
    for pair in inversionPairs.keys():
        print(f"{pair}: {inversionPairs[pair][0]} {inversionPairs[pair][1]} {inversionPairs[pair][2]} {inversionPairs[pair][3]}")

    # Cache the results
    with open(os.path.join(os.path.dirname(__file__), "..", "data", "inversion_pairs.csv"), "w") as f:
        csv_writer = csv.writer(f, delimiter=",")
        csv_writer.writerow(["Card 1", "Card 2", "Card 1 Pick Rate", "Card 2 Pick Rate", "Card 1 GIH Winrate", "Card 2 GIH Winrate"])
        for pair in inversionPairs.keys():
            csv_writer.writerow([pair.split(" & ")[0], pair.split(" & ")[1], inversionPairs[pair][0], inversionPairs[pair][1], inversionPairs[pair][2], inversionPairs[pair][3]])

    return inversionPairs

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
            nextCard.gamesPlayedWinrate = parsePercentage(row[8])

            nextCard.openingHand = float(row[9])
            nextCard.openingHandWinrate = parsePercentage(row[10])

            nextCard.gamesDrawn = int(row[11])
            nextCard.gamesDrawnWinrate = parsePercentage(row[12])

            nextCard.gameInHand = int(row[13])
            nextCard.gameInHandWinrate = parsePercentage(row[14])

            nextCard.gamesNotSeen = int(row[15])
            nextCard.gamesNotSeenWinrate = parsePercentage(row[16])

            # These are like "5.5pp"
            nextCard.improvementWhenDrawn = parsePercentage(row[17])

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

    packCard = getCardFromCardName(packCardName, card_data)
    cardColour = packCard.colour

    for draft in drafts:
        for pick in draft.picks:
            if packCardName not in pick.pack_cards:
                continue

            if packCardName in pick.pack_cards:
                num_colour_cards = 0
                for poolCardName in pick.pool_cards:
                    poolCardData = getCardFromCardName(poolCardName, card_data)
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
    
    if debug == True:
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
    
    if debug == True:
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

    print(f"Elasticity of substitution between {card1} and {card2}: {elasticity}")

    #print(f"Elasticity of substitution between {card2} and {card1}: {elasticity2}")
    return elasticity

def computeDraftStats(cardName1, cardName2, drafts):

    card1 = getCardFromCardName(cardName1, card_data)
    card2 = getCardFromCardName(cardName2, card_data)

    card1Colour = card1.colour
    card2Colour = card2.colour

    card1Seen = 0
    card2Seen = 0
    card1Picked = 0
    card2Picked = 0

    bothCardsSeen = 0
    card1PickedOverCard2 = 0
    card2PickedOverCard1 = 0
    neitherCardPicked = 0

    picksWhereCard1Picked = []
    picksWhereCard2Picked = []


    for draft in drafts:
        for pick in draft.picks:

            # If the pick doesn't contain either card, skip it
            if cardName1 not in pick.pack_cards and cardName2 not in pick.pack_cards:
                continue

            # Pack analysis
            # Count the number of times each card was seen
            # Count the number of times each card was picked
            if cardName1 in pick.pack_cards:
                card1Seen += 1
                if cardName1 == pick.pick:
                    card1Picked += 1
                    picksWhereCard1Picked.append(pick)
            
            if cardName2 in pick.pack_cards:
                card2Seen += 1
                if cardName2 == pick.pick:
                    card2Picked += 1
                    picksWhereCard2Picked.append(pick)

            # Both cards in the pack analysis
            if cardName1 in pick.pack_cards and cardName2 in pick.pack_cards:
                bothCardsSeen += 1
                if cardName1 == pick.pick:
                    card1PickedOverCard2 += 1
                elif cardName2 == pick.pick:
                    card2PickedOverCard1 += 1
                else:
                    neitherCardPicked += 1

            # Pool analysis
            card1InPool = 0
            card2InPool = 0
            coloursInPool = {}
            for colour in COLOURS:
                coloursInPool[colour] = 0

            for poolCardName in pick.pool_cards:
                poolCardColour = getCardFromCardName(poolCardName, card_data).colour
                coloursInPool[poolCardColour] += 1
                if poolCardName == cardName1:
                    card1InPool += 1
                if poolCardName == cardName2:
                    card2InPool += 1

            # Store the pick information in the card objects
            pick.colourInPool = coloursInPool[card1Colour]

            pick.numCardInPool[cardName1] = card1InPool
            pick.numCardInPool[cardName2] = card2InPool

            # Also count packCard as a substitute for itself
            pick.numCardInPool[cardName2] += pick.numCardInPool[cardName1]

            # Store with each card a list of picks where it was seen
            if cardName1 in pick.pack_cards:
                card1.picks.append(pick)
            if cardName2 in pick.pack_cards:
                card2.picks.append(pick)

    card_data[cardName1] = card1
    card_data[cardName2] = card2

    # Set any of these valuse to 0 if the denominator is 0
    if card1Seen == 0:
        card_data[cardName1].pickRate = 0.0
    else:
        card_data[cardName1].pickRate = card1Picked / card1Seen
    if card2Seen == 0:
        card_data[cardName2].pickRate = 0.0
    else:
        card_data[cardName2].pickRate = card2Picked / card2Seen

    # Set the pairwise pick rate for the cards
    if bothCardsSeen == 0:
        card_data[cardName1].pairwisePickRate[cardName2] = 0.0
        card_data[cardName2].pairwisePickRate[cardName1] = 0.0
    else:
        card_data[cardName1].pairwisePickRate[card2] = card1PickedOverCard2 / bothCardsSeen
        card_data[cardName2].pairwisePickRate[card1] = card2PickedOverCard1 / bothCardsSeen

    return card_data
                


# for picks where poth cards were present
# compute the rate at which card 1 was picked
# and the rate at which card 2 was picked
# should sum to 1
def computePairPickRates(pairs, drafts):
    pairsWithPickRates = {}
    # Compute the pairwise pick rate for each pair
    for card1, card2 in pairs:
        pickRate = countSubstitutes(drafts, card1, card2)
        pairsWithPickRates[card1, card2] = pickRate

    # Cache the pairs with their pick rates
    with open(os.path.join(os.path.dirname(__file__), "..", "data", "pick_rates.csv"), "w") as f:
        csv_writer = csv.writer(f, delimiter=",")
        csv_writer.writerow(["Card 1", "Card 2", "Card 1 Pick Rate", "Card 2 Pick Rate"])
        for pair in sortedPairs:
            csv_writer.writerow([pair[0].split(" & ")[0], pair[0].split(" & ")[1], pair[1][0], pair[1][1]])

    return pairsWithPickRates

def checkSubstitution(card1, card2, drafts, card_data, num_multiples=10):
    card_data = computeDraftStats(card1, card2, drafts)

    y = []
    colours = []

    sumCards = {}

    for i in range(num_multiples + 1):
        sumCards[i] = []

    for pick in card_data[card1].picks:
        if pick.pick == card1:
            y.append(1)
        else:
            y.append(0)
        
        colours.append(pick.colourInPool)

        numCard1 = pick.numCardInPool[card1]
        numCard2 = pick.numCardInPool[card2]

        sumOfCards = numCard1 + numCard2
        
        for i in sumCards.keys():
            if sumOfCards == i:
                sumCards[i].append(1)
            else:
                sumCards[i].append(0)


    # Eliminate values that didn't have enough observations
    threshold = 40
    temp = {}
    for i in range(0, num_multiples + 1):
        if sum(sumCards[i]) < threshold:
            print(f"Not enough observations with {i} cards in the pool.")
            print("Will eliminate all higher values")
            break
        else:
            temp[i] = sumCards[i]

    sumCards = temp

    endog = np.array(y)

    exog = 

    for i in sumCards.keys():
        exog = np.column_stack((exog, sumCards[i]))

    # Add a constant term
    exog = sm.add_constant(exog)

    model = sm.OLS(endog, exog)

    results = model.fit()

    print(results.summary())

    

def regressOnNumCard2(card1, card2, drafts, card_data):

    card_data = computeDraftStats(card1, card2, drafts)

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
        debug = True
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

drafts = []
drafts = parseDrafts(csv_file_path, ltr_cards, num_drafts)

card_data = {}
card_data = getCardData()


checkSubstitution("Improvised Club", "Smite the Deathless", drafts, card_data)
exit()

# Count the number of substitutes for Rally at the Hornburg and Smite the Deathless


rallySubs = []
rallyComps = []
smiteSubs = []
smiteComps = []
for cardName in card_data.keys():

    # Only red cards for now
    if card_data[cardName].colour != "R":
        continue

    if regressOnNumCard2(cardName, "Rally at the Hornburg", drafts, card_data):
    #if regression("Rally at the Hornburg", cardName, drafts, card_data):
        rallySubs.append(cardName)
    else:
        rallyComps.append(cardName)
    if regressOnNumCard2(cardName, "Smite the Deathless", drafts, card_data):
    #if regression("Smite the Deathless", cardName, drafts, card_data):
        smiteSubs.append(cardName)
    else:
        smiteComps.append(cardName)

print(f"Number of substitutes for Rally at the Hornburg: {len(rallySubs)}")
print(f"Number of substitutes for Smite the Deathless: {len(smiteSubs)}")

# Print what the substitutes are
print("Substitutes for Rally at the Hornburg:")
for cardName in rallySubs:
    print(cardName)

print("\n========================\n")


print("Substitutes for Smite the Deathless:")
for cardName in smiteSubs:
    print(cardName)

print("\n========================\n")


# For each card, sum the times seen for all its substitutes
print("Number of rally substitutes seen:")
numSeen = 0
for cardName in rallySubs:
    numSeen += card_data[cardName].numberSeen
print(numSeen)

print("Number of smite substitutes seen:")
numSeen = 0
for cardName in smiteSubs:
    numSeen += card_data[cardName].numberSeen
print(numSeen)



exit()



# Generate all the pairs of red cards
redCards = []
for card in card_data.values():
    if card.colour == "R":
        redCards.append(card.name)

# Compute the number of pairs
numPairs = len(redCards) * (len(redCards) - 1) / 2
print(f"Number of pairs: {numPairs}")

# Compute the number of pairs where card2 is a substitute for card1
numSubstitutes = 0
for i in range(0, len(redCards)):
    for j in range(i + 1, len(redCards)):
        card1 = redCards[i]
        card2 = redCards[j]
        if regressOnNumCard2(card1, card2, drafts, card_data):
            numSubstitutes += 1
            card1 = getCardFromCardName(card1, card_data)
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
    card = getCardFromCardName(cardName[0], card_data)
    print(f"{cardName[0]}: {cardName[1]}")
    for substitute in card.substitutes:
        print(f"    {substitute.name}")


# Print the coefficients
exit()


#card_data = computePickRates(drafts, card_data)

# For each colour, generate a list of pairs
# Order doesn't matter
#pairs = getPairs(card_data)

#pairs = computePairPickRates(pairs, drafts)

#inversionPairs = findInversionPairs(pairs, card_data)

# Read in the inversion pairs from the cache
#inversionPairs = readInversionPairsFromCache()





substituteElasticities = []
# Compute the elasticity of substitution for each pair
for cardName in sortedSubstitutes:
    if cardName[1] == 0:
        continue

    card = getCardFromCardName(cardName[0], card_data)
    for substitute in card.substitutes:
        e = elasticitySubstitution(card.name, substitute.name, drafts)

        if e:
            substituteElasticities.append(e)

# Compute the elasticisty of substitution for all pairs

generalElasticities = []
for pair in sortedPairs:
    card1, card2 = getCardsFromPair(pair[0], card_data)
    e = elasticitySubstitution(card1.name, card2.name, drafts)

    if e:
        generalElasticities.append(e)

print(f"Mean elasticity among substitutes: {mean(substituteElasticities)}")
print(f"Mean elasticity among all pairs: {mean(generalElasticities)}")



# Print the time that script took to run
print("Time to run: " + str(time.time() - timestamp))