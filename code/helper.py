OPTIMIZE_STORAGE = True
# Wrapper for picks
class Draft:

    def __init__(self):
        self.draft_id = ""
        self.picks = []


# A draft pick
class Pick:

    def __init__(self):

        self.id = ""

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
        self.openness_to_colour = {}


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

        # We set these during processing
        self.manaValue = 0
        self.cardType = ""

        self.availability_score = 0.0

        self.substitutes = []
        self.timesSeen = 0
        self.timesPicked = 0
        self.picks = []
        self.pairwisePickRate = {}

        # Regression stuff
        self.picked = []
        self.num_sub_in_pool = {}
        self.logified_num_in_pool = {}

        self.complex_subs_in_pool = {}

    # Do sorts by name
    def __lt__(self, other):
        return self.name < other.name

