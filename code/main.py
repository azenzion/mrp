# Import the draft data
import os
import csv

csv_file_path = os.path.join(os.path.dirname(__file__), "..", "data", "draft_data_public.LTR.PremierDraft.csv")
cardlist_file_path = os.path.join(os.path.dirname(__file__), "..", "data", "ltr_cards.txt")

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
        self.pick_maindeck_rate = ""
        self.pick_sideboard_in_rate = ""

        self.pack_cards = []
        self.pool_cards = []



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
        pick.pick_maindeck_rate = row[9]
        pick.pick_sideboard_in_rate = row[10]

        # Parse the cards
        for i in range(11, 277):
            if row[i] == "1":
                pick.pack_cards.append(ltr_cards[i - 11])
        
        # Parse the pool
        for i in range(277, 543):
            if row[i] == "1":
                pick.pool_cards.append(ltr_cards[i - 277])

        # Print the pick and exit
        print("PACK")
        for x, card in enumerate(pick.pack_cards):
            print(f"{x} card")

        print("POOL")
        for card in pick.pool_cards:
            print(card)

        exit()
    # If the pick is part of a draft_id that we already have, add it to that draft
    # Otherwise, create a new draft
    
    if len(drafts) == 0:
        draft = Draft()
        draft.draft_id = pick.draft_id
        draft.picks.append(pick)
        drafts.append(draft)
    else:
        for draft in drafts:
            if draft.draft_id == pick.draft_id:
                draft.picks.append(pick)
                break
        else:

            draft = Draft()
            draft.draft_id = pick.draft_id
            draft.picks.append(pick)
            drafts.append(draft)
