# Import the draft data
import os
import csv
import time

csv_file_path = os.path.join(os.path.dirname(__file__), "..", "data", "draft_data_public.LTR.PremierDraft.csv")
cardlist_file_path = os.path.join(os.path.dirname(__file__), "..", "data", "ltr_cards.txt")

debug = False

# Magic numbers
NUM_HEADERS = 12
END_OF_PACK = 277
END_OF_POOL = 2 * END_OF_PACK - NUM_HEADERS - 1
# Dawn of a new hope is at the end for some reason
DAWN_OF_A_NEW_HOPE_PACK = END_OF_POOL + 1
DAWN_OF_A_NEW_HOPE_POOL = END_OF_POOL + 2

# Parameters
NUM_DRAFTS = 10000000000000000

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


def countSubstitutes(drafts, card1, card2):

    # Look for drafts where a player chose between card1 and card2
    choiceCount = 0
    card1Count = 0
    card2Count = 0
    substituteDrafts = []
    for draft in drafts:
        for pick in draft.picks:
            if card1 in pick.pack_cards and card2 in pick.pack_cards:
                print(f"Pick has {card1} and {card2}")

                # Check if the player picked one of the two cards
                if pick.pick == card1 or pick.pick == card2:
                    print(f"Player picked {pick.pick}")
                    choiceCount += 1
                
                # Check whether they picked card1 or card2
                if pick.pick == card1:
                    card1Count += 1
                if pick.pick == card2:
                    card2Count += 1

                substituteDrafts.append(draft)

    # Print the number of drafts with the choice
    print(f"Number of picks with the choice: {choiceCount}")

    print(f"Number of times they picked {card1}: {card1Count}")
    print(f"Number of times they picked {card2}: {card2Count}")
    
    # Print percentage players chose each card
    print(f"Percentage of times they picked {card1}: {card1Count / choiceCount}")
    print(f"Percentage of times they picked {card2}: {card2Count / choiceCount}")


# Create initial timestamp
timestamp = time.time()

ltr_cards = []
with open(cardlist_file_path, "r") as f:
    for line in f:
        ltr_cards.append(line.strip())

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
        if len(drafts) > NUM_DRAFTS:
            break


countSubstitutes(drafts, "Rally at the Hornburg", "Smite the Deathless")

countSubstitutes(drafts, "Claim the Precious", "Bitter Downfall")

        
# Print the time that script took to run
print("Time to run: " + str(time.time() - timestamp))