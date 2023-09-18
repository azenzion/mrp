# Import the draft data
import os
import csv
import time

csv_file_path = os.path.join(os.path.dirname(__file__), "..", "data", "draft_data_public.LTR.PremierDraft.csv")
cardlist_file_path = os.path.join(os.path.dirname(__file__), "..", "data", "ltr_cards.txt")

debug = False

# Magic numbers
COLOURS = ['W', 'U', 'B', 'R', 'G']
NUM_HEADERS = 12
END_OF_PACK = 277
END_OF_POOL = 2 * END_OF_PACK - NUM_HEADERS - 1
# Dawn of a new hope is at the end for some reason
DAWN_OF_A_NEW_HOPE_PACK = END_OF_POOL + 1
DAWN_OF_A_NEW_HOPE_POOL = END_OF_POOL + 2

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

    # Do sorts by name
    def __lt__(self, other):
        return self.name < other.name

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
                if row[i] == "1":
                    pick.pool_cards.append(ltr_cards[i - END_OF_PACK])

            # Dawn of a new hope
            if row[DAWN_OF_A_NEW_HOPE_PACK] == "1":
                pick.pack_cards.append("Dawn of a New Hope")
            if row[DAWN_OF_A_NEW_HOPE_POOL] == "1":
                pick.pool_cards.append("Dawn of a New Hope")

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

            # Stop after 1000 drafts
            if len(drafts) > numDrafts:
                break
    return drafts

def findInversionPairs(pairs, card_data):
    inversionPairs = {}
    # Narrow down to pairs where the card with the lower GIH winrate is picked more often
    for pair in pairs.keys():
        card1Name = pair.split(" & ")[0]
        card2Name = pair.split(" & ")[1]

        card1 = None
        card2 = None
        for card in card_data:
            if card.name == card1Name:
                card1 = card
            if card.name == card2Name:
                card2 = card

        if not card1 or not card2:
            print("ERROR: Card not found")
            print(f"Card 1: {card1Name}")
            print(f"Card 2: {card2Name}")
            break
        
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
    card_data = []
    with open(os.path.join(os.path.dirname(__file__), "..", "data", "ltr-card-ratings-2023-09-17.csv"), "r") as f:
        csv_reader = csv.reader(f, delimiter=",")

        header_row = next(csv_reader)
        print(header_row)

        next(csv_reader)


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

            card_data.append(nextCard)
    return card_data

def getPairs(card_data):
    for colour in COLOURS:
        for card in card_data:
            if card.colour == colour:
                for otherCard in card_data:
                    if otherCard.colour == colour:
                        if card != otherCard:
                            pair = [card, otherCard]
                            pair.sort()
                            pairName = pair[0].name + " & " + pair[1].name
                            if pairName not in pairs.keys():
                                pairs[pairName] = 0

    # Print the number of pairs
    print(f"Number of pairs: {len(pairs.keys())}")  
    return pairs

def computePairPickRates(pairs, drafts):
    # Compute the pairwise pick rate for each pair
    for pair in pairs.keys():
        card1 = pair.split(" & ")[0]
        card2 = pair.split(" & ")[1]
        pickRate = countSubstitutes(drafts, card1, card2)
        pairs[pair] = pickRate

    # Cache the pairs with their pick rates
    with open(os.path.join(os.path.dirname(__file__), "..", "data", "pick_rates.csv"), "w") as f:
        csv_writer = csv.writer(f, delimiter=",")
        csv_writer.writerow(["Card 1", "Card 2", "Card 1 Pick Rate", "Card 2 Pick Rate"])
        for pair in sortedPairs:
            csv_writer.writerow([pair[0].split(" & ")[0], pair[0].split(" & ")[1], pair[1][0], pair[1][1]])

# Create initial timestamp
timestamp = time.time()

ltr_cards = []
with open(cardlist_file_path, "r") as f:
    for line in f:
        ltr_cards.append(line.strip())

drafts = parseDrafts(csv_file_path, ltr_cards, 1000000)

card_data = getCardData()

# For pairs of cards, compute the likelihood that a player will pick one over the other

# For each colour, generate a list of pairs
# Order doesn't matter
pairs = getPairs(card_data)

#pairs = computePairPickRates(drafts, pairs)

#inversionPairs = findInversionPairs(pairs, card_data)

# Read in the pair pick rates from the cache
with open(os.path.join(os.path.dirname(__file__), "..", "data", "pick_rates.csv"), "r") as f:
    csv_reader = csv.reader(f, delimiter=",")
    next(csv_reader)
    for row in csv_reader:
        pairName = row[0] + " & " + row[1]
        pairs[pairName] = [float(row[2]), float(row[3])]

# Read in the inversion pairs from the cache
inversionPairs = {}
with open(os.path.join(os.path.dirname(__file__), "..", "data", "inversion_pairs.csv"), "r") as f:
    csv_reader = csv.reader(f, delimiter=",")
    next(csv_reader)
    for row in csv_reader:
        pairName = row[0] + " & " + row[1]
        inversionPairs[pairName] = [float(row[2]), float(row[3]), float(row[4]), float(row[5])]

# Print the pairs sorted by pick rate
sortedPairs = sorted(pairs.items(), key=lambda x: x[1][0], reverse=True)

for pair in sortedPairs:
    print(f"{pair[0]}: {pair[1][0]} {pair[1][1]}")



# Print the time that script took to run
print("Time to run: " + str(time.time() - timestamp))